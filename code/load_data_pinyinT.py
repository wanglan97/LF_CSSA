import csv
import random
import time

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from config import config
import pickle
import os
from tqdm import tqdm
# from config import config

class MMdataset(Dataset):
    def __init__(self, index, mode='train'):
        self.index = index
        self.mode = mode
        self.__init_msaZH()

    def __init_msaZH(self):
        data = np.load('E:\\audio\chsims\\features\data_pyT.npz',allow_pickle=True)
        self.text = data['feature_T'][self.index[self.mode]]
        self.audio =data['feature_A'][self.index[self.mode]]
        self.pinyin =data['feature_PY'][self.index[self.mode]]

        self.label = {
            'A': data['label_A'][self.index[self.mode]]
        }
        self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)

        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.audio[self.audio != self.audio] = 0

        self.audio = np.transpose(self.audio, (1, 0, 2))
    def __len__(self):
        return len(self.index[self.mode])

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'pinyin': torch.Tensor(self.pinyin[index]),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.label.items()}

        }
        return sample

    def get_seq_len(self):  # text 39,audio 400,vision 55,pinyin 36
        return (self.text.shape[1], self.audio.shape[1],self.pinyin.shape[1])

    def get_feature_dim(self):  # text 768,audio 33,vision 709
        return (self.text.shape[2], self.audio.shape[2],self.pinyin.shape[2])
    def get_max_len(self):
        return (self.text.shape[0],self.audio.shape[0])


def MMdataloader(config):
#100%
    # test_index = np.array(pd.read_csv(r'E:\audio\chsims\label\test_index.csv')).reshape(-1)  # 457
    # train_index = np.array(pd.read_csv(r'E:\audio\chsims\label\train_index.csv')).reshape(-1)   # 1368
    # val_index = np.array(pd.read_csv(r'E:\audio\chsims\label\val_index.csv')).reshape(-1)   # 456

    # 60%
    test_index = np.array(pd.read_csv(r'E:\audio\chsims\label\60%-test_index.csv')).reshape(-1)  # 274
    train_index = np.array(pd.read_csv(r'E:\audio\chsims\label\60%-train_index.csv')).reshape(-1)  # 820
    val_index = np.array(pd.read_csv(r'E:\audio\chsims\label\60%-val_index.csv')).reshape(-1)  # 273
#80%

    # test_index = np.array(pd.read_csv(r'E:\audio\chsims\label\80%-test_index.csv')).reshape(-1)  # 365
    # train_index = np.array(pd.read_csv(r'E:\audio\chsims\label\80%-train_index.csv')).reshape(-1)  # 1094
    # val_index = np.array(pd.read_csv(r'E:\audio\chsims\label\80%-val_index.csv')).reshape(-1)  # 364

    print('Train Samples Num:{0}'.format(len(train_index)))
    print('Valid Samples Num:{0}'.format(len(val_index)))
    print('Test Samples Num:{0}'.format(len(test_index)))

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }
    datasets = {
        'train': MMdataset(index=index, mode='train'),
        'valid': MMdataset(index=index, mode='valid'),
        'test': MMdataset(index=index, mode='test'),
    }

    #print(datasets['train'].__getitem__(1367)['labels']['T'])#label_T.csv 1068
    # print(datasets['train'].get_feature_dim())
    print(datasets['train'].get_seq_len())
    dataLoader = {
       ds: DataLoader(datasets[ds], batch_size=config.batch_size, shuffle=True)
       for ds in datasets.keys()
    }
    return dataLoader
# from config.config import Config
# config=Config()
# dataloader=MMdataloader(config)
