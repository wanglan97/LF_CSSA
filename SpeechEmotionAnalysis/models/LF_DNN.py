"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from FeatureNets import SubNet,TextSubNet

class LF_DNN(nn.Module):
    """
    late fusion using DNN
    """
    def __init__(self, args):
        super(LF_DNN, self).__init__()
        self.text_in, self.audio_in = args.feature_dims
        self.text_hidden, self.audio_hidden = args.hidden_dims

        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.audio_hidden, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, args.output_dim)


    def forward(self, text_x, audio_x):
        audio_x = audio_x.squeeze(1)
        audio_h = self.audio_subnet(audio_x)
        text_h = self.text_subnet(text_x)

        fusion_h = torch.cat([audio_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)
        return output
