import torch
import math
import torch.nn as nn

def masked_fill_inf(x, mask):
    # mask: True = valid, False = masked
    x = x.masked_fill(~mask, float('-inf'))
    return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        k是[batch_size, head, length, d_k]的四维张量
        """
        batch_size, head, length, d_k = k.size()

        k_t = k.transpose(2, 3)
        score = torch.matmul(q, k_t) / math.sqrt(d_k)

        if mask is not None:
            score = masked_fill_inf(score, mask)

        score = self.softmax(score)

        v = torch.matmul(score, v)

        return v, score