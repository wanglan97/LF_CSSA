"""
获取pikle文件特征，用在了train_cnn中
"""

import librosa.display
# 数据科学工具包
import numpy as np
# import tensorflow as tf
# 以递归方式查找所有的音频文件
import os
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
# print(labels[:10])
"""获取文本特征"""
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


def getTextFeature(df):
    model = KeyedVectors.load_word2vec_format('RawData/CASIA database/text20211223.model.bin', binary=True)
    index2word_set = set(model.index_to_key)

    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    # feature=avg_feature_vector("你们 称呼 长辈",model,300,index2word_set)
    sentence = 0
    with open("RawData/CASIA database/text_jieba.txt", encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            feature = avg_feature_vector(line, model, 300, index2word_set)
            # print('feature', feature)
            # df.loc[sentence,'text_feature'] = feature
            # print(feature)
            df.loc[sentence] = [feature]
            sentence = sentence + 1


df1 = pd.DataFrame(columns=['feature'])
getTextFeature(df1)
df1 = pd.DataFrame(df1['feature'].values.tolist())
# print('df1',df1)
"""
获取音频特征
"""
df2 = pd.DataFrame(columns=['feature'])
bookmark = 0
for index, y in enumerate(list_name):
    # 音频导入函数
    # X : 音频的信号值，类型是ndarray,sample_rate : 采样率
    # y 音频路径|sr 采样率（默认22050，但是有重采样的功能）
    # duration 获取音频的时长 |offset 音频读取的时间
    # https://www.cnblogs.com/xingshansi/p/6816308.html
    X, sample_rate = librosa.load(y, res_type='kaiser_fast'
                                  , duration=2.5, sr=22050 * 2, offset=0.5)
    # 得到采样率
    sample_rate = np.array(sample_rate)
    # MFCC提取：
    # https://www.jianshu.com/p/24044f4c3531
    # 并且求取平均值
    f0 = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512).T, axis=0)  # (seq_len, 1),计算音频时间序列的过零率。
    cqt = np.mean(librosa.feature.chroma_cqt(y=y, sr=22050 * 2, hop_length=512).T, axis=0)  # (seq_len, 12)
    mfccs = np.mean(librosa.feature.mfcc(y=X,
                                         sr=sample_rate,
                                         n_mfcc=13),
                    axis=0)
    # 获取音频特征
    feature = np.concatenate([f0, mfccs, cqt], axis=0)
    # print('feature',feature)
    df2.loc[bookmark] = [feature]
    bookmark = bookmark + 1
df2 = pd.DataFrame(df2['feature'].values.tolist())
# print('df2', df2)
# 获取音频特征取值

# df3 = pd.DataFrame(df['feature'].values.tolist())  # 216
# df4 = pd.DataFrame(df['text_feature'].values.tolist())  # 300
# print('df3',df3)
# print(df4)
# 音频特征、文本特征和目标特征合并
newdf = pd.concat([df1, df2, labels], axis=1)
df3 = pd.DataFrame(newdf.values.tolist())
# print('df3',df3)
# 将旧的名字label转换成新的名字，0
# rnewdf = newdf.rename(index=str, columns={"0": "label"})
# print(newdf[:])
print("拼接后特征维度", df3.shape)

# 将特征打乱
from sklearn.utils import shuffle

rnewdf = shuffle(newdf)
# print(rnewdf[:10])
# 将其中的NAN填充成0
rnewdf = rnewdf.fillna(0)
# 随机生成数组
newdf1 = np.random.rand(len(rnewdf)) < 0.8
# 获取训练集
train = rnewdf[newdf1]
# 获取测试集
test = rnewdf[~newdf1]
# print(train[250:260])

# 获取训练特征
trainfeatures = train.iloc[:, :-1]
print('trainfeatures', trainfeatures.shape)
audio_train = train.iloc[:, 300:516]
print('audio_train', audio_train.shape)
text_train = train.iloc[:, 0:300]
print('text_train', text_train.shape)
# print(audiofeatures)
# print('text_train',text_train)
# 获取训练标签
trainlabel = train.iloc[:, -1:]
# 获取测试特征
testfeatures = test.iloc[:, :-1]
audio_test = test.iloc[:, 300:516]
text_test = test.iloc[:, 0:300]
# 获取测试标签
testlabel = test.iloc[:, -1:]
print('训练集样本数：', len(trainfeatures))
print('测试集样本数：', len(testfeatures))
# print('train_feature',trainfeatures)
# print('train_label', trainlabel)
# print('testfeature',testfeatures)
# print('test_label',testlabel)

# 转成onehot
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

import torch.utils.data as Data

X_train = np.array(trainfeatures)  # 961,216
audio_train = np.array(audio_train)
print('audio_train', audio_train.shape)
text_train = np.array(text_train)
audio_test = np.array(audio_test)
text_test = np.array(text_test)

y_train = np.array(trainlabel)  # 934

# print('text_train', text_train.shape)

X_test = np.array(testfeatures)  # 223,216

y_test = np.array(testlabel)  # 266

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
# print('train_label',y_train)
# print('X_train', X_train.shape)
# print('X_test', X_test.shape)
# print('y_train:', y_train)
# print('y_test:', y_test.shape)
# #增加一个维度
x_traincnn = np.expand_dims(X_train, axis=2)
# 增加一个维度
x_testcnn = np.expand_dims(X_test, axis=2)
# print(x_traincnn.shape)#(样本数，216，1）
# print(x_testcnn.shape)

# 将数据保存到pikle

import pickle

data = {'audio_train': audio_train,
        'text_train': text_train,
        'audio_test': audio_test,
        'text_test': text_test,
        'y_train': y_train,
        'y_test': y_test
        }
#
# selfref_list = [1, 2, 3]
# selfref_list.append(selfref_list)

output = open('data/dataWithText.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(data, output)

# Pickle the list using the highest protocol available.
# pickle.dump(selfref_list, output, -1)
output.close()
