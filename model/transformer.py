import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_voc_size, tgt_voc_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, voc_size=src_voc_size, max_len=max_len)
        self.decoder = Decoder(num_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, voc_size=tgt_voc_size, max_len=max_len)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src_seq, src_mask)
        output = self.decoder(tgt_seq, enc_output, tgt_mask, memory_mask)
        return output 