"""
此代码可以忽略
"""
import os

import librosa


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
# print(list_name)   #1200个语音文件


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
import numpy as np
labels = pd.DataFrame(label_list)

"""
获取音频特征
"""
df = pd.DataFrame(columns=['feature'])
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
    f0 = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512).T,axis=0) # (seq_len, 1),计算音频时间序列的过零率。
    cqt = np.mean(librosa.feature.chroma_cqt(y=y, sr=22050 * 2, hop_length=512).T,axis=0) # (seq_len, 12)
    mfccs = np.mean(librosa.feature.mfcc(y=X,
                                         sr=sample_rate,
                                         n_mfcc=13),
                    axis=0)
    # 获取音频特征
    feature = np.concatenate([f0, mfccs, cqt], axis=0)
    # print('feature',feature)
    df.loc[bookmark] = [feature]
    bookmark = bookmark + 1
# print(df)
# 获取音频特征取值
df3 = pd.DataFrame(df['feature'].values.tolist())
# print('df3:',df3)
# 音频特征和目标特征合并
newdf = pd.concat([df3, labels], axis=1)#1200,217
print('newdf',newdf.shape)


# 将特征打乱
from sklearn.utils import shuffle
rnewdf = shuffle(newdf)
# print(rnewdf[:10])
# 将其中的NAN填充成0
rnewdf=rnewdf.fillna(0)
# 随机生成数组
newdf1 = np.random.rand(len(rnewdf)) < 0.8
# 获取训练集
train = rnewdf[newdf1]
# 获取测试集
test = rnewdf[~newdf1]
# print(train[250:260])

# 获取训练特征
trainfeatures = train.iloc[:, :-1]
# print('trainfeatures:',trainfeatures)
#获取训练标签
trainlabel = train.iloc[:, -1:]
#获取测试特征
testfeatures = test.iloc[:, :-1]
# 获取测试标签
testlabel = test.iloc[:, -1:]
# print(len(trainfeatures))
# print(len(testfeatures))
# print('train_feature',trainfeatures)
# print('train_label',trainlabel)
# print('testfeature',testfeatures)
# print('test_label',testlabel)

#转成onehot
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
# print('train_1',trainfeatures)
X_train = np.array(trainfeatures)#961,216
# print('train_2',trainfeatures)
y_train = np.array(trainlabel) #934

X_test = np.array(testfeatures)#223,216

y_test = np.array(testlabel)#266

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
# print('train_label',y_train)
print(X_train.shape)
print(X_test.shape)