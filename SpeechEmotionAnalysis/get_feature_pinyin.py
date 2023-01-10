"""
获取特征
"""
import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob

from gensim.models import KeyedVectors
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import *


def listdir(path, list_name):
    #     list_name=list()
    label_list = list()
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.wav':
            list_name.append(file_path)


list_name = list()
listdir('RawData/CASIA database', list_name)
class getFeatures():
    def __init__(self, working_dir, pretrainedBertPath):
        self.data_dir = working_dir
        # padding
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        # toolkits path
        self.pretrainedBertPath = pretrainedBertPath

    def __getTextEmbedding(self, text):
        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        pretrained_weights = self.pretrainedBertPath
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        # add_special_tokens will add start and end token
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze().numpy()
    def word2idx(self,pylist):
        wvmodel = KeyedVectors.load_word2vec_format('RawData/CASIA database/pyT20220108.model.bin', binary=True)
        self.vocab_size = len(wvmodel)  # 字典大小 338
        self.embed_size = wvmodel.vector_size  # 字向量维度300
        temp=torch.zeros(1,self.embed_size,dtype=torch.float)
        zeros=np.zeros(self.embed_size,dtype=np.float)
        for item in pylist:
            try:
                wordvec=wvmodel[item]
            except KeyError:
                wordvec=zeros
            temp=torch.cat((temp,torch.from_numpy(wordvec).float().unsqueeze(dim=0)),dim=0)
        print("每句的拼音的长度:",temp.size())
        return temp
    def __getPinyinWithoutTEmbedding(self,pinyin):

        # weights = torch.FloatTensor(wvmodel.vectors)
        # self.weights = nn.Embedding.from_pretrained(weights).weight
        # self.embedding = nn.Embedding(self.vocab_size, self.embed_size, _weight=self.weights)
        embeddings=self.word2idx(pinyin)
        return embeddings.numpy()

    def __getPinyinWithTEmbedding(self, pinyin):
        wvmodel = KeyedVectors.load_word2vec_format('RawData/CASIA database/pyT20220108.model.bin', binary=True)
        self.vocab_size = len(wvmodel)  # 字典大小 338
        self.embed_size = wvmodel.vector_size  # 字向量维度300
        temp = torch.zeros(1, self.embed_size, dtype=torch.float)
        zeros = np.zeros(self.embed_size, dtype=np.float)
        for item in pinyin:
            try:
                wordvec = wvmodel[item]
            except KeyError:
                wordvec = zeros
            temp = torch.cat((temp, torch.from_numpy(wordvec).unsqueeze(dim=0)), dim=0)
        print("每句的拼音的长度:", temp.size())
        return temp.numpy()



    def __getAudioEmbedding(self, audio_path):
        y, sr = librosa.load(audio_path)
        # print('sr=',sr)

        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512  # hop_length smaller, seq_len larger,  hop_length ：S列之间的音频样本数,帧移
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T  # (seq_len, 1),计算音频时间序列的过零率。
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T  # (seq_len, 12)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T  # (seq_len, 20)

        return np.concatenate([f0, mfcc, cqt], axis=-1)  # (seq_len, 33)

    def __padding(self, feature, MAX_LEN):
        """
        mode:
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        assert self.padding_mode in ['zeros', 'normal']
        assert self.padding_location in ['front', 'back']

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]

        if self.padding_mode == "zeros":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif self.padding_mode == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN - length, feature.shape[1]))

        feature = np.concatenate([pad, feature], axis=0) if (self.padding_location == "front") else \
            np.concatenate((feature, pad), axis=0)
        return feature

    def __paddingSequence(self, sequences):
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]
        # confirm length using (mean + std)
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence


    def get_labels(self):
        # print(len(list_name))   #1200个语音文件

        label_list = list()
        angry_num = 0
        fear_num = 0
        for i in range(len(list_name)):
            name = list_name[i]
            if "angry" in name:
                label_list.append("angry")
                angry_num += 1
            #     else:
            #         print("nothing")
            if "fear" in name:
                label_list.append("fear")
                fear_num += 1
            #     else:
            #         print("nothing")
            if "happy" in name:
                label_list.append("happy")
            #     else:
            #         print("nothing")
            if "neutral" in name:
                label_list.append("neutral")
            #     else:
            #         print("nothing")
            if "sad" in name:
                label_list.append("sad")
            #     else:
            #         print("nothing")
            if "surprise" in name:
                label_list.append("surprise")
            # else:
            #     print("nothing")
        # print(angry_num,fear_num)
        import pandas as pd

        labels = pd.DataFrame(label_list)
        labels=np.array(labels)
        lb = LabelEncoder()
        labels = np_utils.to_categorical(lb.fit_transform(labels))
        return labels
    def results(self, output_dir,labels):

        features_T, features_A, features_pinyin = [], [], []
        text = []
        pinyinT = []
        with open("RawData/CASIA database/pinyinWithT.txt", encoding='utf8') as f:
            for line in f:
                line = line.strip('\n')
                pinyinT.append(line)
        with open("RawData/CASIA database/text.txt", encoding='utf8') as f:
            for line in f:
                line = line.strip('\n')
                text.append(line)

        #提取语音特征

        for i,audio in enumerate(list_name):
            # text
            embedding_T = self.__getTextEmbedding(text[i])
            pinyin=pinyinT[i]
            pinyin=pinyin.split()
            # embedding_PY = self.__getPinyinWithoutTEmbedding(pinyin)
            embedding_PY = self.__getPinyinWithTEmbedding(pinyin)
            # features_TandPY.append()
            features_T.append(embedding_T)
            features_pinyin.append(embedding_PY)
            # audio

            embedding_A = self.__getAudioEmbedding(audio)
            features_A.append(embedding_A)
            # labels
        # padding
        feature_T = self.__paddingSequence(features_T)
        features_A = self.__paddingSequence(features_A)
        features_pinyin = self.__paddingSequence(features_pinyin)

        # save
        # save_path = os.path.join(self.data_dir, output_dir, 'data_py.npz')
        train_save_path = os.path.join(self.data_dir, output_dir, 'data_pyT_train.npz')
        test_save_path = os.path.join(self.data_dir, output_dir, 'data_pyT_test.npz')
        np.savez(train_save_path, \
                 feature_T=feature_T[:900], feature_A=features_A[:900], feature_PY=features_pinyin[:900],
                 label_A=labels[:900])
        np.savez(test_save_path, \
                 feature_T=feature_T[900:], feature_A=features_A[900:], feature_PY=features_pinyin[900:],
                 label_A=labels[900:])

        print('Features are saved in %s!' % train_save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='path to CH-SIMS')
    parser.add_argument('--openface2Path', type=str,
                        help='path to FeatureExtraction tool in openface2')
    parser.add_argument('--pretrainedBertPath', type=str,
                        help='path to pretrained bert directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # data_dir = '/path/to/MSA-ZH'
    # gf = getFeatures(args.data_dir, args.openface2Path, args.pretrainedBertPath)
    gf = getFeatures("G:\speech\SpeechEmotionAndPeopleAnalyse-main",
                     "D:\Installpackage\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12")
    labels=gf.get_labels()
    # print(labels.shape[0])
    # train_label=labels[:900]
    # print(train_label.shape[0])
    # gf.handleImages()

    gf.results('data',labels)

    # test=np.load('data.npz')
    # print(len(test['feature_V']))
