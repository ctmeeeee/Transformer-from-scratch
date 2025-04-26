import torch
import torch.nn as nn
from model.decoder_block import DecoderBlock
from positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, voc_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, d_model)
        self.linear = nn.Linear(d_model, voc_size)
        self.pos_encoding = PositionalEncoding(d_model, device='cuda', max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_ff, n_heads) for _ in range(num_layers)
        ])

    def forward(self, enc_output, tgt_seq, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt_seq)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        x = self.linear(x)
        return x

