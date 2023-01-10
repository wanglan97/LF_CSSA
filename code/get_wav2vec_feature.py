"""
获得语音转录文本特征特征
"""
import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob

import torchaudio
from gensim.models import KeyedVectors
from tqdm import tqdm
import torch
from transformers import *


class getFeatures():
    def __init__(self, working_dir, pretrainedBertPath):
        self.data_dir = working_dir
        self.label_path = os.path.join(working_dir, 'label')
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


    def __getAudioEmbedding(self, audio_path):
        y, sr = librosa.load(audio_path)
        # print('sr=',sr)

        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512  # hop_length smaller, seq_len larger,  hop_length ：S列之间的音频样本数,帧移
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T  # (seq_len, 1),计算音频时间序列的过零率。
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T  # (seq_len, 12)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T  # (seq_len, 20)

        return np.concatenate([f0, mfcc, cqt], axis=-1)  # (seq_len, 33)

    def __getPinyinWithTEmbedding(self, pinyin):
        wvmodel = KeyedVectors.load_word2vec_format('./data/pinyinT20211127.model.bin', binary=True)
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
        # print("每句的拼音的长度:", temp.size())
        return temp.numpy()
    def __getWav2vec(self, audio_path):
        audio, rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(rate, 16_000)
        audio = resampler(audio).squeeze().numpy()
        # print(audio.shape)
        # print(audio)
        # print(rate)
        processor = Wav2Vec2Processor.from_pretrained("./pretrained_model/")
        model = Wav2Vec2ForCTC.from_pretrained("./pretrained_model/")
        input1 = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        # 获取logit值(非规范化值)
        input2 = input1.input_values.squeeze()
        # print(input2.size())
        with torch.no_grad():
            logit = model(input2).logits
        # print(logit.shape)  # [1, 267, 21128]  #帧数
        prediction = torch.argmax(logit, dim=-1)
        transcription = processor.batch_decode(prediction)[0]
        from pypinyin import pinyin, lazy_pinyin, Style
        s = lazy_pinyin('我不喜欢这个电影', style=Style.TONE3, neutral_tone_with_five=True)
        s = " ".join(i for i in s)
        return s,transcription
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

    def results(self, output_dir):
        features_trans=[]
        features_pinyin=[]

        for root, dirs, files in os.walk("E:\\audio\\chsims\\audio"):
            for file in files:
                # print(os.path.join(root, file))
                path = os.path.join(root, file)
                # print(path)
                embedding_pinyin,embedding_trans=self.__getWav2vec(path)
                embedding_trans = self.__getTextEmbedding(embedding_trans)
                embedding_pinyin=self.__getPinyinWithTEmbedding(embedding_pinyin)
                # features_wav.append(embedding_wav)
                features_pinyin.append(embedding_pinyin)
                features_trans.append(embedding_trans)

        # padding
        features_trans = self.__paddingSequence(features_trans)
        features_pinyin = self.__paddingSequence(features_pinyin)
        # save
        save_path = os.path.join(self.data_dir, output_dir, 'data_wav2vec1.npz')
        np.savez(save_path, \
                 feature_trans=features_trans,feature_pinyin=features_pinyin
                 )

        print('Features are saved in %s!' % save_path)




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
    gf = getFeatures("E:\\audio\\chsims",
                     "D:\Installpackage\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12")

    # gf.handleImages()

    gf.results('features')

    # test=np.load('data.npz')
    # print(len(test['feature_V']))
