# """
# 转录
# """
import numpy as np
import wave
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#
#
wavFile = r"E:\LFCSSA\SIMS\audio\0001.wav"
# wavFile = r"F:\speech_analysis\powerful_chinese_ASR-main\powerful_chinese_ASR-main\0039.wav"   #我票在气候婆现在可以了妙吗公司六安排的凯几别
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
# print(audio.shape)
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





