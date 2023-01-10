#使用pikle

import pickle

import torch
from torch.utils.data import DataLoader, Dataset


class MMdataset(Dataset):
    def __init__(self, mode='train'):

        self.mode = mode
        self.__init_msaZH()

    def __init_msaZH(self):
        pkl_file = open('../data/dataWithPYT.pkl', 'rb')
        # pkl_file = open('data/dataWithPYT.pkl', 'rb')
        data = pickle.load(pkl_file)
        if(self.mode=='train'):
            self.audio = data['audio_train']
            self.pyT=data['pyT_train']
            self.text = data['text_train']
            self.label = data['y_train']
        else:
            self.audio = data['audio_test']
            self.text = data['text_test']
            self.pyT = data['pyT_test']
            self.label = data['y_test']
        pkl_file.close()

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
        return (self.text.shape[1], self.audio.shape[1],self.pyT.shape[1])
    def get_seq_len(self):  # text 768,audio 33,vision 709
        return (self.text.shape[0], self.audio.shape[0],self.pyT.shape[0])

def MMdataloader():

    datasets = {
        'train': MMdataset(mode='train'),
        'test': MMdataset(mode='test'),
    }

    #print(datasets['train'].__getitem__(1367)['labels']['T'])#label_T.csv 1068
    print(datasets['train'].get_feature_dim())
    print(datasets['train'].get_seq_len())
    dataLoader = {
       ds: DataLoader(datasets[ds], batch_size=32, shuffle=True)
       for ds in datasets.keys()
    }
    return dataLoader

dataloader=MMdataloader()
# print(dataloader)   arg.batch_size