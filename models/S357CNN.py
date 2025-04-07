import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


class S357CNN(nn.Module):
    def __init__(self, in_features, kernel_num, kernel_sizes, CNN_dropout):
        super(S357CNN, self).__init__()
        self.in_features = in_features
        self.Co = kernel_num
        self.Ks = kernel_sizes

        self.convK_1 = nn.ModuleList([nn.Conv1d(in_features, self.Co, K, padding=K // 2) for K in self.Ks])

        self.CNN_dropout = nn.Dropout(CNN_dropout)

        self.fc = nn.Linear(len(kernel_sizes) * self.Co, in_features)

    def forward(self, features):

        x = features.permute(0, 2, 1)

        conv_out = [F.relu(conv(x)) for conv in self.convK_1]

        out = torch.cat(conv_out, dim=1)
        out = self.CNN_dropout(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out)
        return out


