import torch
import torch.nn as nn
from multihead_attention import MultiheadAttention
from layer_norm import AddNorm
from positionwise_ffn import PositionWiseFFN

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiheadAttention(d_model, n_heads)
        self.norm1 = AddNorm(d_model, dropout)
        self.FFN = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x, self.attention(x, x, x, mask))
        x = self.norm2(x, self.FFN(x))
        return x