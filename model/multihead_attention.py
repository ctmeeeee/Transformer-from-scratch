import torch
import math
import torch.nn as nn
from scaled_dotproduct_attention import ScaledDotProductAttention


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiheadAttention, self).__init__()
        '''
        n_heads: 注意力头数
        d_model: 模型维度(通常为512)
        '''
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        #1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        #2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        #3. do scale dot product to compute similarity

        out, attention = self.attention(q, k, v, mask=mask)

        #4. concat and pass linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    # 将输入的QKV根据注意力头数进行划分
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_heads
        tensor = tensor.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)

        return tensor

    # 最后将各个头的输出进行拼接
    def concat(self, tensor):
        #print(tensor.size())
        batch_size, head, length, d_tensor = tensor.size()

        d_model = d_tensor * head

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor