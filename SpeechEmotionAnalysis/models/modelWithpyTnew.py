import torch
import torch.nn as nn
from torch.nn import functional as F

"""
文本情感标签作为语音的辅助标签，多任务学习，基于LSTM的模型，我们的模型
"""
import torch.nn as nn
import torch.nn.functional as F


class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_size = [128, 64, 64]
        self.audio_out = 64
        self.text_out = 64
        self.dropout = [0, 0.1, 0]
        self.output_dim = 6
        self.text_feature_dim = 768
        self.pyT_feature_dim = 300
        self.audio_feature_dim = 33
        self.audio_linear = 64
        self.text_linear = 128
        # self.pyT_linear=128
        self.audio_subnet = SubNet(self.audio_feature_dim, self.hidden_size[1], self.dropout[1])
        self.text_subnet = SubNet(self.text_feature_dim, self.hidden_size[0], self.dropout[0])
        self.pyT_subnet = SubNet(self.pyT_feature_dim, self.hidden_size[2], self.dropout[2])
        # self.text_subnet = TextSubNet(self.text_feature_dim, self.hidden_size[0], self.text_out, 1, dropout=self.dropout[0])
        # self.pyT_subnet = TextSubNet(self.pyT_feature_dim, self.hidden_size[2], self.text_out, 1, dropout=self.dropout[2])
        self.dropout_linear_t = nn.Dropout(p=self.dropout[0])
        self.dropout_linear_a = nn.Dropout(p=self.dropout[1])
        self.post_fusion_layer_1 = nn.Linear(self.hidden_size[0] + self.hidden_size[2], self.text_linear)
        self.post_fusion_layer_2 = nn.Linear(self.text_linear, self.output_dim)
        self.post_fusion_layer_3 = nn.Linear(self.hidden_size[1] + self.output_dim, self.audio_linear)
        self.post_fusion_layer_4 = nn.Linear(self.audio_linear, self.output_dim)

    def forward(self, text_x, audio_x, pyT_x):
        audio_x = audio_x.squeeze()
        text_x = text_x.squeeze()
        pyT_x = pyT_x.squeeze()
        batch_size = text_x.data.shape[0]
        # print('batch_size',batch_size)
        text_h = self.text_subnet(text_x)  # 64
        pyT_h = self.pyT_subnet(pyT_x)  # 64
        text_h = text_h.squeeze()
        text_h = text_h.view(batch_size, -1)
        pyT_h = pyT_h.squeeze()
        pyT_h = pyT_h.view(batch_size, -1)
        fusion = torch.cat((text_h, pyT_h), dim=1)
        t_out = F.relu(self.post_fusion_layer_1(self.dropout_linear_t(fusion)))
        t_out = self.post_fusion_layer_2(t_out)
        audio_h = self.audio_subnet(audio_x)  # 16
        audio_h = audio_h.squeeze()
        audio_h = audio_h.view(batch_size, -1)
        # print(audio_h.size(),t_out.size())
        input = torch.cat((audio_h, t_out), dim=1)
        a_out = F.relu(self.post_fusion_layer_3(self.dropout_linear_a(input)))
        a_out = self.post_fusion_layer_4(a_out)
        return t_out, a_out
