# from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import PointWiseFeedForward


def hopfield_update_full(teacherK, studentQ, beta=1.0, alpha=0.1, gamma_update=0.5, diag_reg=True):
    B, L, d = teacherK.shape
    t2d = teacherK.reshape(B * L, d)
    s2d = studentQ.reshape(B * L, d)

    KQ = t2d @ s2d.T
    
    attn = F.softmax(beta*KQ, dim=1)
    align = attn @ s2d

    reg = alpha * t2d
    if diag_reg:
        row_norm = torch.sum(t2d * t2d, dim=1)
        half_norm = 0.5*row_norm
        row_sm = torch.softmax(half_norm, dim=0)
        diag_mat = torch.diag(row_sm)
        diag = diag_mat @ t2d
    else:
        diag = torch.zeros_like(t2d)

    updated = t2d + gamma_update * (align - reg - diag)
    return updated.reshape(B, L, d)


class _HopBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads)
        self.ffn = PointWiseFeedForward(hidden, dropout)

    def forward(self, x: torch.Tensor):
        x = self.ln(x.transpose(0, 1))
        attn_out, _ = self.attn(x, x, x)
        out = x + attn_out
        return self.ffn(out.transpose(0, 1))

class FlipBlockST(nn.Module):
    def __init__(self, hidden, heads, dropout, *, beta, alpha, gamma, num_hopblock=2):
        super().__init__()
        self.student_layers = nn.ModuleList([
            _HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)
        ])
        self.teacher_layers = nn.ModuleList([
            _HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)
        ])
        self.cfg = dict(beta=beta, alpha=alpha, gamma_update=gamma)

    def forward(self, collab, semantic):
        s, t = collab, semantic
        for layer_s, layer_t in zip(self.student_layers, self.teacher_layers):
            s = layer_s(s)
            t = layer_t(t)
        t_upd = hopfield_update_full(t, s, **self.cfg)
        return s, t_upd

class FlipBlockCT(nn.Module):
    def __init__(self, hidden, heads, dropout, *, beta, alpha, gamma, num_hopblock=2):
        super().__init__()
        self.teacher_layers = nn.ModuleList([
            _HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)
        ])
        self.student_layers = nn.ModuleList([
            _HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)
        ])
        self.cfg = dict(beta=beta, alpha=alpha, gamma_update=gamma)

    def forward(self, collab, semantic):
        t, s = collab, semantic
        for layer_t, layer_s in zip(self.teacher_layers, self.student_layers):
            t = layer_t(t)
            s = layer_s(s)
        t_upd = hopfield_update_full(t, s, **self.cfg)
        return t_upd, s


from models.Dualview_model import (
    DualViewSASRec,
    DualViewGRU4Rec
)

_ENCODER = {
    "sasrec": DualViewSASRec,
    "gru4rec": DualViewGRU4Rec
}

def _encoder(user_num, item_num, device, args):
    key = getattr(args, "encoder", "sasrec").lower()
    if getattr(args, "model_name", "").lower().startswith("fcrllm_"):
        key = args.model_name.split("_", 1)[1]
    if key not in _ENCODER:
        raise ValueError(f"Unknown encoder '{key}'.")
    return _ENCODER[key](user_num, item_num, device, args)


class _BaseFlip(nn.Module):
    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.enc = _encoder(user_num, item_num, device, args)
        self.hid = args.hidden_size
        self.gamma_fc = args.gamma_fc
        self.pos_emb = self.enc.pos_emb
        self.do = self.enc.emb_dropout

    def _views(self, seq, pos):
        pos = pos.long()
        co = self.enc.id_item_emb(seq) * (self.hid ** 0.5)
        co = self.do(co + self.pos_emb(pos))
        se = self.enc.adapter(self.enc.llm_item_emb(seq)) * (self.hid ** 0.5)
        se = self.do(se + self.pos_emb(pos))
        return co, se

    def predict(self, seq, item_indices, positions, **kw):
        return self.enc.predict(seq, item_indices, positions, **kw)

    def get_user_emb(self, seq, positions, **kw):
        return self.enc.get_user_emb(seq, positions, **kw)

class FlipModelST(_BaseFlip):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.block = FlipBlockST(
            self.hid, args.num_heads, args.dropout_rate,
            beta=args.beta_hopfield, alpha=args.alpha_hopfield,
            gamma=args.gamma_update, num_hopblock=args.num_hopblock
        )
        self.mse = nn.MSELoss()

    def forward(self, seq, pos, neg, positions, **kw):
        rank_loss = self.enc(seq, pos, neg, positions, **kw)

        co, se = self._views(seq, positions)
        stud, teach_upd = self.block(co, se)
        align_loss = self.mse(stud, teach_upd)

        return rank_loss + self.gamma_fc * align_loss


class FlipModelCT(_BaseFlip):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.block = FlipBlockCT(
            self.hid, args.num_heads, args.dropout_rate,
            beta=args.beta_hopfield, alpha=args.alpha_hopfield,
            gamma=args.gamma_update, num_hopblock=args.num_hopblock
        )
        self.mse = nn.MSELoss()

    def forward(self, seq, pos, neg, positions, **kw):
        rank_loss = self.enc(seq, pos, neg, positions, **kw)

        co, se = self._views(seq, positions)
        teach_upd, stud = self.block(co, se)
        align_loss = self.mse(teach_upd, stud)

        return rank_loss + self.gamma_fc * align_loss

class FlipClassEnsemble(nn.Module):
    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.hyb_alpha = args.hyb_alpha
        self.st = FlipModelST(user_num, item_num, device, args)
        self.ct = FlipModelCT(user_num, item_num, device, args)

    def forward(self, seq, pos, neg, positions, **kw):
        loss_st = self.st(seq, pos, neg, positions, **kw)
        loss_ct = self.ct(seq, pos, neg, positions, **kw)
        
        return self.hyb_alpha * loss_st + (1 - self.hyb_alpha) * loss_ct

    
    @torch.no_grad()
    def predict(self, seq, item_indices=None, items=None, positions=None, **kw):
        if item_indices is None:
            item_indices = items
        logits_st = self.st.predict(seq, item_indices, positions, **kw)
        logits_ct = self.ct.predict(seq, item_indices, positions, **kw)
        return self.hyb_alpha * logits_st + (1 - self.hyb_alpha) * logits_ct

    @torch.no_grad()
    def get_user_emb(self, seq, positions, **kw):
        e_st = self.st.get_user_emb(seq, positions, **kw)
        e_ct = self.ct.get_user_emb(seq, positions, **kw)
        return self.hyb_alpha * e_st + (1 - self.hyb_alpha) * e_ct