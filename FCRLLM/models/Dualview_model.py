import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq


def _load_pretrained_embedding(path: str, pad_size: int = 1) -> torch.Tensor:
    emb = pickle.load(open(path, "rb"))
    emb = np.insert(emb, 0, values=np.zeros((1, emb.shape[1])), axis=0)
    emb = np.concatenate([emb, np.zeros((pad_size, emb.shape[1]))], axis=0)
    return torch.tensor(emb, dtype=torch.float32)


def _build_adapter_linear(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2),
        nn.Linear(in_dim // 2, out_dim),
    )


class _DualViewMixin:
    def _init_dual_embeddings(self, args):
        llm_path = os.path.join("data", args.dataset, "handled", "itm_emb_np.pkl")
        id_path = os.path.join("data", args.dataset, "handled", "pca64_itm_emb_np.pkl")

        self.llm_item_emb = nn.Embedding.from_pretrained(
            _load_pretrained_embedding(llm_path)
        )
        if getattr(args, "freeze", False):
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.adapter = _build_adapter_linear(
            self.llm_item_emb.embedding_dim, args.hidden_size
        )

        self.id_item_emb = nn.Embedding.from_pretrained(
            _load_pretrained_embedding(id_path)
        )
        self.id_item_emb.weight.requires_grad = True

        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        if hasattr(self, "_init_weights"):
            self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
            self._init_weights()

    def _get_embedding(self, items: torch.Tensor) -> torch.Tensor:
        id_emb = self.id_item_emb(items)
        llm_emb = self.adapter(self.llm_item_emb(items))
        return torch.cat([id_emb, llm_emb], dim=-1)

    def log2feats(self, seq, pos=None):
        id_seq = self.id_item_emb(seq) * (self.id_item_emb.embedding_dim ** 0.5)
        llm_seq = self.adapter(self.llm_item_emb(seq)) * (self.id_item_emb.embedding_dim ** 0.5)

        if pos is not None:
            pos_emb = self.pos_emb(pos.long())
            id_seq += pos_emb
            llm_seq += pos_emb

        id_seq = self.emb_dropout(id_seq)
        llm_seq = self.emb_dropout(llm_seq)

        id_feats = self.backbone(id_seq, seq)
        llm_feats = self.backbone(llm_seq, seq)

        return torch.cat([id_feats, llm_feats], dim=-1)

    @torch.no_grad()
    def predict(self, seq, item_indices, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        return (item_embs @ final_feat.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def get_user_emb(self, seq, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        return log_feats[:, -1, :]


class DualViewGRU4Rec(_DualViewMixin, GRU4Rec):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_dual_embeddings(args)


class DualViewSASRec(_DualViewMixin, SASRec_seq):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_dual_embeddings(args)
