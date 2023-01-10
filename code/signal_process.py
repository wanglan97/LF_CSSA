"""
获取语音的基本参数
汉字转拼音、分词、转录
语音信号处理：重采样、预加重、分帧、加窗

"""


#读取语音、获取属性信息
# import wave
# wavFile = r"E:\audio\data_thchs30\data_thchs30\train\A32_210.wav"
# f = wave.open(wavFile)
# # 音频头 参数
# params = f.getparams()
# Channels = f.getnchannels()
# SampleRate = f.getframerate()
# bit_type = f.getsampwidth() * 8
# frames = f.getnframes()
# # Duration 也就是音频时长 = 采样点数/采样率
# Duration = wav_time = frames / float(SampleRate)  # 单位为s
#
# print("音频头参数：", params)
# print("通道数(Channels)：", Channels)
# print("采样率(SampleRate)：", SampleRate)
# print("比特(Precision)：", bit_type)
# print("采样点数(frames)：", frames)
# print("帧数或者时间(Duration)：", Duration)


# 汉字转拼音
from pypinyin import pinyin, lazy_pinyin, Style
# s=lazy_pinyin('我不喜欢这个电影', style=Style.TONE3, neutral_tone_with_five=True)
# s1=lazy_pinyin('我不喜欢这个电影')
# print(s)
# s=" ".join(i for i in s)
# s1=" ".join(i for i in s1)
# print(s)
# print(s1)
#
# with open('./data/word.txt',encoding='utf8') as f:
#     for line in f:
#         line=line.strip('\n')
#         line=line.split(' ')[1]
#         # pinyin=lazy_pinyin(line, style=Style.TONE3, neutral_tone_with_five=True)
#         # pinyin=" ".join(i for i in pinyin)
#         # with open('./data/pinyinWithT.txt','a',encoding='utf8') as fw:
#         #     fw.write(pinyin)
#         #     fw.write('\n')
#         pinyinlazy=lazy_pinyin(line)
#         pinyinlazy = " ".join(i for i in pinyinlazy)
#         with open('./data/pinyinWithoutT.txt','a',encoding='utf8') as fw1:
#             fw1.write(pinyinlazy)
#             fw1.write('\n')
#         # print(pinyin)
#


# 分词
# import jieba as jieba
# s="我不喜欢这个电影"
# s=jieba.cut(s)
# s=" ".join(s)
# print(s)
# with open('./data/word.txt',encoding='utf8') as f:
#     for line in f:
#         line = line.strip('\n')
#         index=line.split(' ')[0]
#         line=line.split(' ')[1]
#         line=" ".join(jieba.cut(line))
#         with open('./data/word_jieba.txt', 'a', encoding='utf8') as fw1:
#             line=index+" "+line
#             fw1.write(line)
#             fw1.write('\n')

# s="ai you mei shen me bu hao yi si de ni shi yi ge zhi de tuo fu de ren"
# pinyin=s.split()
# print(pinyin)
#

"""
转录
"""
"""
import wave
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# wavFile = r"E:\audio\data_thchs30\data_thchs30\train\A11_0.wav"
wavFile = r"F:\speech_analysis\powerful_chinese_ASR-main\powerful_chinese_ASR-main\0039.wav"   #我票在气候婆现在可以了妙吗公司六安排的凯几别
# f = wave.open(wavFile, "rb")
# str_data = f.readframes(f.getnframes())
# f.close()
# file = wave.open(wavFile, 'wb')
# file.setnchannels(1)
# file.setsampwidth(4)
# file.setframerate(16000)
# file.writeframes(str_data)
# file.close()
# audio, rate = librosa.load(wavFile)
audio, rate = torchaudio.load(wavFile)
# # print(audio.shape)
resampler = torchaudio.transforms.Resample(rate, 16_000)
audio = resampler(audio).squeeze().numpy()
# # print(audio.shape)
# print(audio)
# print(rate)
processor = Wav2Vec2Processor.from_pretrained("./pretrained_model/")
model = Wav2Vec2ForCTC.from_pretrained("./pretrained_model/")
input1 = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
# 获取logit值(非规范化值)
input2=input1.input_values.squeeze()
# # print(input2.size())
with torch.no_grad():
    logit = model(input2).logits
print(logit.shape)  #[1, 267, 21128]  #帧数
prediction = torch.argmax(logit, dim=-1)
# print(prediction.size()) #1,267
# 最后一步是将预测传递给分词器解码以获得转录
transcription = processor.batch_decode(prediction)[0]
# Printing the transcription
print(transcription)

"""
"""
librosa读取，重采样、预加重、分帧、加窗

# """
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
#读取原始音频
# y,sr=librosa.load("G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Raw\\video_0001\\0022.mp4",sr=48000)
y,sr=librosa.load("G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Processed\\audio\\video_0001\\0022.wav",sr=48000)
y_16k=librosa.resample(y,sr,16000)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.tick_params(direction='in')
# librosa.output.write_wav(y_16k.mp4,y_16k,16000)
# librosa.display.waveplot(y,sr=48000,color="black")
# librosa.display.waveplot(y_16k,sr=16000)


# plt.xlabel('时间（s）',fontproperties=font)
# plt.ylabel('振幅',fontproperties=font)
# plt.title('分离音频',fontproperties=font)
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.show()
#
#
# # 预加重
def pre_fun(x):  # 定义预加重函数
    signal_points=len(x)  # 获取语音信号的长度
    signal_points=int(signal_points)  # 把语音信号的长度转换为整型
    # s=x  # 把采样数组赋值给函数s方便下边计算
    for i in range(1, signal_points, 1):# 对采样数组进行for循环计算
        x[i] = x[i] - 0.95 * x[i - 1]  # 一阶FIR滤波器
    return x  # 返回预加重以后的采样数组
pre_emphasis = pre_fun(y_16k)  # 函数调用
# librosa.display.waveplot(pre_emphasis,sr=16000)
# plt.xlabel('时间（s）',fontproperties=font)
# plt.ylabel('振幅',fontproperties=font)
# plt.title('预加重',fontproperties=font)
# plt.show()
#
#
#
#
# '''分帧'''
# import numpy as np
import librosa.display  # 导入音频及绘图显示包
import matplotlib.pyplot as plt  # 导入绘图工作的函数集合

# 分帧
# def frame(x, lframe, mframe):  # 定义分帧函数
#     signal_length = len(x)  # 获取语音信号的长度
#     fn = (signal_length-lframe)/mframe  # 分成fn帧
#     fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
#     fn1 = int(fn1)  # 将帧数化为整数
#     # 求出添加的0的个数
#     numfillzero = (fn1*mframe+lframe)-signal_length
#     # 生成填充序列
#     fillzeros = np.zeros(numfillzero)
#     # 填充以后的信号记作fillsignal
#     fillsignal = np.concatenate((x,fillzeros))  # concatenate连接两个维度相同的矩阵
#     # 对所有帧的时间点进行抽取，得到fn1*lframe长度的矩阵d
#     d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1*mframe, mframe), (lframe, 1)).T
#     # 将d转换为矩阵形式（数据类型为int类型）
#     d = np.array(d, dtype=np.int32)
#     signal = fillsignal[d]
#     return(signal, fn1, numfillzero)
# lframe = int(sr*0.03)  # 帧长(持续0.025秒)
# mframe = int(sr*0.015)  # 帧移
# # 函数调用，把采样数组、帧长、帧移等参数传递进函数frame，并返回存储于endframe、fn1、numfillzero中
# endframe, fn1, numfillzero = frame(pre_emphasis, lframe, mframe)

# 显示第1帧波形图
# x1 = np.arange(0, lframe, 1)  # 第1帧采样点刻度
# x2 = np.arange(0, lframe/sr, 1/sr)  # 第1帧时间刻度
# 显示波形图
# plt.figure()
# plt.plot(x1, endframe[0])
# plt.xlabel('points')  # x轴
# plt.ylabel('wave')  # y轴
# plt.title('bluesky1 firstframe  wave', fontsize=12, color='black')
# plt.show()
# plt.figure()
# plt.plot(x2, endframe[0])
# plt.xlabel('时间（s）',fontproperties=font)  # x轴
# plt.ylabel('波形',fontproperties=font)  # y轴
# plt.title('分帧', fontproperties=font)
# plt.show()
#
#
'''加窗'''
import numpy as np
import librosa.display  # 导入音频及绘图显示包
import matplotlib.pyplot as plt  # 导入绘图工作的函数集合

# 分帧
def frame(x, lframe, mframe):  # 定义分帧函数
    signal_length = len(x)  # 获取语音信号的长度
    fn = (signal_length-lframe)/mframe  # 分成fn帧
    fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
    fn1 = int(fn1)  # 将帧数化为整数
    # 求出添加的0的个数
    numfillzero = (fn1*mframe+lframe)-signal_length
    # 生成填充序列
    fillzeros = np.zeros(numfillzero)
    # 填充以后的信号记作fillsignal
    fillsignal = np.concatenate((x,fillzeros))  # concatenate连接两个维度相同的矩阵
    # 对所有帧的时间点进行抽取，得到fn1*lframe长度的矩阵d
    d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1*mframe, mframe), (lframe, 1)).T
    # 将d转换为矩阵形式（数据类型为int类型）
    d = np.array(d, dtype=np.int32)
    signal = fillsignal[d]
    return(signal, fn1, numfillzero)
lframe = int(sr*0.03)  # 帧长(持续0.025秒)
mframe = int(sr*0.015)  # 帧移
# 函数调用，把采样数组、帧长、帧移等参数传递进函数frame，并返回存储于endframe、fn1、numfillzero中
endframe, fn1, numfillzero = frame(y, lframe, mframe)

# 对第一帧进行加窗
hanwindow = np.hanning(lframe)  # 调用汉明窗，把参数帧长传递进去
signalwindow = endframe[0]*hanwindow  # 第一帧乘以汉明窗
x1 = np.arange(0, lframe, 1)  # 第一帧采样点刻度
x2 = np.arange(0, lframe/sr, 1/sr)  # 第一帧时间刻度
# plt.figure()
plt.plot(x2, signalwindow)
plt.xlabel('时间（s）',fontproperties=font)  # x轴
plt.ylabel('波形',fontproperties=font)  # y轴
plt.title('加窗', fontproperties=font)
plt.show()
