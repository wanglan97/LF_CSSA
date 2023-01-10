"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
from AlignNets import AlignSubNet

# from models.EF_LSTM import EF_LSTM
# from models.LMF import LMF
from models.Graph_MFN import Graph_MFN
# from model.Graph_MFN import Graph_MFN
__all__ = ['AMIO']



class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_align = args.need_align
        text_seq_len, _ = args.input_len
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_align):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            args.input_len = self.alignNet.get_seq_len()

        self.Model = Graph_MFN(args)

    def forward(self, text_x, audio_x):
        if(self.need_align):
            text_x, audio_x = self.alignNet(text_x, audio_x)
        return self.Model(text_x, audio_x)

