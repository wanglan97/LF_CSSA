import torch
import torch.nn as nn
from torch.nn import functional as F
"""
仅用声音进行情感分类
"""
import torch.nn as nn
import torch.nn.functional as F

class SubNet(nn.Module):
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
        super(SubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.hidden_size=hidden_size
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        #64,39,768
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze()) #32,64,64
        y_1 = self.linear_1(h)
        return y_1

class AudioSubNet(nn.Module):
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
        super(AudioSubNet, self).__init__()
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



class Model(nn.Module):  #with normalize
    def __init__(self):
        super(Model, self).__init__()

        self.hidden_size = 64
        self.dropout = 0
        self.output_dim = 6
        self.audio_feature_dim = 229
        self.audio_linear=64
        self.audio_subnet = AudioSubNet(self.audio_feature_dim, self.hidden_size, self.dropout)
        self.dropout_linear_a=nn.Dropout(p=self.dropout)
        self.post_fusion_layer_3 = nn.Linear(self.hidden_size, self.audio_linear)
        self.post_fusion_layer_4 = nn.Linear(self.audio_linear, self.output_dim)

    def forward(self, text_x, audio_x):
        batch_size = text_x.data.shape[0]
        audio_h = audio_x.squeeze()
        audio_h = self.audio_subnet(audio_h)  # 16

        audio_h = audio_h.view(batch_size, -1)
        a_out = F.relu(self.post_fusion_layer_3(self.dropout_linear_a(audio_h)))
        a_out = self.post_fusion_layer_4(a_out)
        return a_out,a_out
