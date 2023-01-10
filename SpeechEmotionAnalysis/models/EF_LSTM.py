"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EF_LSTM(nn.Module):
    """
    early fusion using lstm
    """
    def __init__(self, args):
        super(EF_LSTM, self).__init__()
        text_in, audio_in = args.feature_dim
        in_size = text_in + audio_in
        input_len = args.input_len
        hidden_size = args.hidden_dim
        dropout = args.dropout
        output_dim = args.output_dim
        num_layers = args.num_layers
        self.norm = nn.BatchNorm1d(input_len)
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, text_x, audio_x):
        # early fusion (default: seq_l == seq_a == seq_v)
        x = torch.cat([text_x, audio_x], dim=-1)
        # print(x.size())
        x = self.norm(x)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze())
        x = F.relu(self.linear(x), inplace=True)#inplace=true,直接对上一层传下来的值进行修改
        x = self.dropout(x)
        output = self.out(x)
        return output
        
class EF_CNN(nn.Module):
    """
    early fusion using cnn
    """
    def __init__(self, args):
        super(EF_CNN, self).__init__()

    def forward(self, text_x, audio_x, video_x):
        pass