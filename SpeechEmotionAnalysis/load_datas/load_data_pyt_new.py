#使用npz
import pickle
"""
是由get_feature_pinyin生成的


"""
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
class MMdataset(Dataset):
    def __init__(self, config,mode='train'):
        self.config=config
        self.mode = mode
        self.__init_msaZH()

    def __init_msaZH(self):
        train_data = np.load('../data/data_pyT_train.npz',allow_pickle=True)
        test_data = np.load('../data/data_pyT_test.npz',allow_pickle=True)
        # pkl_file = open('data/dataWithPYT.pkl', 'rb')

        if(self.mode=='train'):
            self.audio = train_data['feature_A']
            self.pyT=train_data['feature_PY']
            self.text = train_data['feature_T']
            self.label = train_data['label_A']
        else:
            self.audio = test_data['feature_A']
            self.pyT = test_data['feature_PY']
            self.text = test_data['feature_T']
            self.label = test_data['label_A']
        if(self.config.need_normalized):
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)

        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.text = np.transpose(self.text, (1, 0, 2))
        self.pyT = np.transpose(self.pyT, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.text = np.mean(self.text, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.pyT = np.mean(self.pyT, axis=0, keepdims=True)

        # remove possible NaN values
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0
        self.pyT[self.pyT != self.pyT] = 0

        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.text = np.transpose(self.text, (1, 0, 2))
        self.pyT = np.transpose(self.pyT, (1, 0, 2))
        print(self.text.shape)
    def __len__(self):
            return len(self.label)
    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'pyT': torch.Tensor(self.pyT[index]),
            'audio': torch.Tensor(self.audio[index]),
            'label': torch.Tensor(self.label[index])

        }
        return sample

    def get_feature_dim(self):  # text 768,audio 33,vision 709
        return (self.text.shape[2], self.audio.shape[2],self.pyT.shape[2])
    def get_seq_len(self):  # text 768,audio 33,vision 709
        return (self.text.shape[1], self.audio.shape[1],self.pyT.shape[1])

def MMdataloader(config):

    datasets = {
        'train': MMdataset(config,mode='train'),
        'test': MMdataset(config,mode='test'),
    }

    #print(datasets['train'].__getitem__(1367)['labels']['T'])#label_T.csv 1068
    print(datasets['train'].get_feature_dim())
    print(datasets['train'].get_seq_len())
    dataLoader = {
       ds: DataLoader(datasets[ds], batch_size=config.batch_size, shuffle=True)
       for ds in datasets.keys()
    }
    return dataLoader

# dataloader=MMdataloader()
# print(dataloader)   arg.batch_size