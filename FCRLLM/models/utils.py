import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.linear1  = nn.Linear(hidden_units, hidden_units * 4)
        self.act      = self.gelu
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2  = nn.Linear(hidden_units * 4, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, inputs) -> torch.Tensor:
        x = self.linear1(inputs)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + inputs)
        
        return x

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
