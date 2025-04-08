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

        # 多尺度卷积层，使用1D卷积操作来处理输入序列
        self.convK_1 = nn.ModuleList([nn.Conv1d(in_features, self.Co, K, padding=K // 2) for K in self.Ks])

        self.CNN_dropout = nn.Dropout(CNN_dropout)

        # 线性层确保输出维度与输入维度一致
        self.fc = nn.Linear(len(kernel_sizes) * self.Co, in_features)

    def forward(self, features):
        # features: [batch_size, seq_len, in_features] => [32, 128, 768]
        x = features.permute(0, 2, 1)  # 转换维度以适应 Conv1d [batch_size, in_features, seq_len]

        # 多尺度卷积
        conv_out = [F.relu(conv(x)) for conv in self.convK_1]
        # Concatenate over the channel dimension (i.e., hidden size)
        out = torch.cat(conv_out, dim=1)  # [batch_size, Co*len(Ks), seq_len]
        out = self.CNN_dropout(out)
        out = out.permute(0, 2, 1)  # 转换回 [batch_size, seq_len, Co*len(Ks)]
        out = self.fc(out)  # 确保输出维度与输入维度一致 [batch_size, seq_len, in_features]

        return out


#
# def main():
#     # 定义超参数
#     batch_size = 32
#     seq_len = 128
#     in_features = 768  # 输入特征维度
#     kernel_num = 100  # 卷积核数量
#     kernel_sizes = [3, 5, 7]  # 不同的卷积核大小
#     cnn_dropout = 0.5
#
#     # 初始化模型
#     model = TC_base(in_features, kernel_num, kernel_sizes, cnn_dropout)
#
#     # 创建一个随机输入张量
#     input_data = torch.randn(batch_size, seq_len, in_features)
#
#     # 前向传播
#     output = model(input_data)
#
#     # 输出输入和输出的维度
#     print("Input shape:", input_data.shape)
#     print("Output shape:", output.shape)
#
# if __name__ == "__main__":
#     main()
