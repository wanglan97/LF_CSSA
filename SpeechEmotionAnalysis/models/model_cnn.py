"""
仅用cnn和语音
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         # self.batch_size = config.batch_size
#         self.conv1 = nn.Conv1d(in_channels=216, out_channels=256, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
#         self.dropout=nn.Dropout(0.1)
#         self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
#         self.fc = nn.Linear(in_features=128, out_features=6)
#     def forward(self,x):
#         print(x.size())
#         out=F.relu(self.conv1(x))
#         out=self.dropout(F.relu(self.conv2(out)))
#         # out=self.maxpool(out)
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out=out.view(-1,out.size(1))
#         fc_out=self.fc(out)
#         return fc_out
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
#基于CNN融合文本和拼音
# class Model(nn.Module):
#  def __init__(self):
#     super(Model, self).__init__()
#     # self.batch_size = config.batch_size
#     self.conv1 = nn.Conv1d(in_channels=229, out_channels=256, kernel_size=1)
#     self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
#     self.dropout=nn.Dropout(0.1)
#     self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2)
#     self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
#     self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
#     self.fc = nn.Linear(in_features=128, out_features=6)
#     self.batch_size = 32
#     self.hidden_size = [64,64,64]
#     self.audio_out = 64
#     self.text_out = 64
#     self.dropout1 = [0,0,0]
#     self.output_dim = 6
#     self.text_feature_dim = 300
#     self.pyT_feature_dim = 300
#     self.audio_feature_dim = 229
#     self.audio_linear=64
#     self.text_linear=128
#     # self.pyT_linear=128
#     self.audio_subnet = SubNet(self.audio_feature_dim, self.hidden_size[1],self.dropout1[1])
#     self.text_subnet = SubNet(self.text_feature_dim, self.hidden_size[0],self.dropout1[0])
#     self.pyT_subnet = SubNet(self.pyT_feature_dim, self.hidden_size[2],self.dropout1[2])
#     self.dropout_linear_t=nn.Dropout(p=self.dropout1[0])
#     self.dropout_linear_a=nn.Dropout(p=self.dropout1[1])
#     self.post_fusion_layer_1 = nn.Linear(self.hidden_size[0]+self.hidden_size[2], self.text_linear)
#     self.post_fusion_layer_2 = nn.Linear(self.text_linear, self.output_dim)
#     self.post_fusion_layer_3 = nn.Linear(128+self.output_dim, self.audio_linear)
#     self.post_fusion_layer_4 = nn.Linear(self.audio_linear, self.output_dim)
#  def forward(self,text_x, audio_x,pyT_x):
#     # print(audio_x.size())
#     audio_x=audio_x.unsqueeze(dim=2)
#     out=F.relu(self.conv1(audio_x))
#     out=self.dropout(F.relu(self.conv2(out)))
#     # out=self.maxpool(out)
#     out = F.relu(self.conv3(out))
#     out = F.relu(self.conv4(out))
#     out=out.view(-1,out.size(1))
#     # fc_out=self.fc(out)
#     batch_size = text_x.data.shape[0]
#     # print('batch_size',batch_size)
#     text_h = self.text_subnet(text_x)  # 64
#     pyT_h = self.pyT_subnet(pyT_x)  # 64
#     text_h = text_h.squeeze()
#     text_h = text_h.view(batch_size, -1)
#     pyT_h = pyT_h.squeeze()
#     pyT_h = pyT_h.view(batch_size, -1)
#     fusion = torch.cat((text_h, pyT_h), dim=1)
#     t_out = F.relu(self.post_fusion_layer_1(self.dropout_linear_t(fusion)))
#     t_out = self.post_fusion_layer_2(t_out)
#     # print(audio_h.size(),t_out.size())
#     input = torch.cat((out, t_out), dim=1)
#     a_out = F.relu(self.post_fusion_layer_3(self.dropout_linear_a(input)))
#     a_out = self.post_fusion_layer_4(a_out)
#     return t_out,a_out

#cnn融合文本
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=229, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.fc = nn.Linear(in_features=128, out_features=6)
        self.hidden_size = [64, 64]
        self.audio_out = 64
        self.text_out = 64
        self.dropout1 = [0.1, 0]
        self.output_dim = 6
        self.text_feature_dim = 300
        self.audio_feature_dim = 229
        self.audio_linear = 128
        self.text_linear = 64
        # self.pyT_linear=128
        self.audio_subnet = SubNet(self.audio_feature_dim, self.hidden_size[1], self.dropout1[1])
        self.text_subnet = SubNet(self.text_feature_dim, self.hidden_size[0], self.dropout1[0])
        self.dropout_linear_t = nn.Dropout(p=self.dropout1[0])
        self.dropout_linear_a = nn.Dropout(p=self.dropout1[1])
        self.post_fusion_layer_1 = nn.Linear(self.hidden_size[0], self.text_linear)
        self.post_fusion_layer_2 = nn.Linear(self.text_linear, self.output_dim)
        self.post_fusion_layer_3 = nn.Linear(128 + self.output_dim, self.audio_linear)
        self.post_fusion_layer_4 = nn.Linear(self.audio_linear, self.output_dim)

    def forward(self, text_x, audio_x):
        audio_x = audio_x.unsqueeze(dim=2)
        out = F.relu(self.conv1(audio_x))
        out = self.dropout(F.relu(self.conv2(out)))
        # out=self.maxpool(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(-1, out.size(1))
        # fc_out=self.fc(out)
        batch_size = text_x.data.shape[0]
        # print('batch_size',batch_size)
        text_h = self.text_subnet(text_x)  # 64

        text_h = text_h.squeeze()
        text_h = text_h.view(batch_size, -1)
        t_out = F.relu(self.post_fusion_layer_1(self.dropout_linear_t(text_h)))
        t_out = self.post_fusion_layer_2(t_out)
        # print(audio_h.size(),t_out.size())
        input = torch.cat((out, t_out), dim=1)
        a_out = F.relu(self.post_fusion_layer_3(self.dropout_linear_a(input)))
        a_out = self.post_fusion_layer_4(a_out)
        return t_out,a_out