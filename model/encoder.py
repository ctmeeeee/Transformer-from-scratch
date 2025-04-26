import torch
import torch.nn as nn
from encoder_block import EncoderBlock
from positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, voc_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, device='cuda', max_len=max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_ff, n_heads) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x