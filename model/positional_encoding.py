import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len):
        '''

        参数d_model: 模型维度
        参数max_len: 最大序列长度
        参数device: 设备
        '''
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        #从一维张量到二维，方便后续运算进行广播操作，变成[max_len, d_model/2]的矩阵

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        '''
        参数x: [batch_size, seq_len, d_model]的张量
        '''
        seq_len = x.size(1)
        pos_enc = self.encoding[:seq_len, :].unsqueeze(0)
        return x + pos_enc