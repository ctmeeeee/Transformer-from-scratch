import torch
import torch.nn as nn
from positionwise_ffn import PositionWiseFFN
from multihead_attention import MultiheadAttention
from layer_norm import AddNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_atten = MultiheadAttention(d_model, n_heads)
        self.enc_dec_atten = MultiheadAttention(d_model, n_heads)
        self.norm1 = AddNorm(d_model, dropout)
        self.norm2 = AddNorm(d_model, dropout)
        self.norm3 = AddNorm(d_model, dropout)
        self.FFN = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.norm1(x, self.self_atten(x, x, x, tgt_mask))
        x = self.norm2(x, self.enc_dec_atten(x, enc_output, enc_output, memory_mask))
        x = self.norm3(x, self.FFN(x))
        return x