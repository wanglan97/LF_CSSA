"""
用于测试的代码，可忽略
"""
import numpy as np
import wave

import math

"""
def nextpow2(n):
    '''
    求最接近数据长度的2的整数次方
    An integer equal to 2 that is closest to the length of the data

    Eg:
    nextpow2(2) = 1
    nextpow2(2**10+1) = 11
    nextpow2(2**20+1) = 21
    '''
    return np.ceil(np.log2(np.abs(n))).astype('long')

# 打开WAV文档
f = wave.open('parrots/data/0002.wav')
# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
fs = framerate
# 读取波形数据
str_data = f.readframes(nframes)
f.close()
# 将波形数据转换为数组
x = np.fromstring(str_data, dtype=np.short)
# 计算参数
len_ = 20 * fs // 1000  # 样本中帧的大小
PERC = 50  # 窗口重叠占帧的百分比
len1 = len_ * PERC // 100  # 重叠窗口
len2 = len_ - len1  # 非重叠窗口
# 设置默认参数
Thres = 3
Expnt = 2.0
beta = 0.002
G = 0.9
# 初始化汉明窗
win = np.hamming(len_)
# normalization gain for overlap+add with 50% overlap
winGain = len2 / sum(win)

# Noise magnitude calculations - assuming that the first 5 frames is noise/silence
nFFT = 2 * 2 ** (nextpow2(len_))
noise_mean = np.zeros(nFFT)

j = 0
for k in range(1, 6):
    noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
    j = j + len_
noise_mu = noise_mean / 5

# --- allocate memory and initialize various variables
k = 1
img = 1j
x_old = np.zeros(len1)
Nframes = len(x) // len2 - 1
xfinal = np.zeros(Nframes * len2)

# =========================    Start Processing   ===============================
for n in range(0, Nframes):
    # Windowing
    insign = win * x[k - 1:k + len_ - 1]
    # compute fourier transform of a frame
    spec = np.fft.fft(insign, nFFT)
    # compute the magnitude
    sig = abs(spec)

    # save the noisy phase information
    theta = np.angle(spec)
    SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)


    def berouti(SNR):
        if -5.0 <= SNR <= 20.0:
            a = 4 - SNR * 3 / 20
        else:
            if SNR < -5.0:
                a = 5
            if SNR > 20:
                a = 1
        return a


    def berouti1(SNR):
        if -5.0 <= SNR <= 20.0:
            a = 3 - SNR * 2 / 20
        else:
            if SNR < -5.0:
                a = 4
            if SNR > 20:
                a = 1
        return a


    if Expnt == 1.0:  # 幅度谱
        alpha = berouti1(SNRseg)
    else:  # 功率谱
        alpha = berouti(SNRseg)
    #############
    sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt;
    # 当纯净信号小于噪声信号的功率时
    diffw = sub_speech - beta * noise_mu ** Expnt


    # beta negative components

    def find_index(x_list):
        index_list = []
        for i in range(len(x_list)):
            if x_list[i] < 0:
                index_list.append(i)
        return index_list


    z = find_index(diffw)
    if len(z) > 0:
        # 用估计出来的噪声信号表示下限值
        sub_speech[z] = beta * noise_mu[z] ** Expnt
        # --- implement a simple VAD detector --------------
    if SNRseg < Thres:  # Update noise spectrum
        noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
        noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
    # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
    # 交换上下对称元素
    sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
    x_phase = (sub_speech ** (1 / Expnt)) * (
                np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
    # take the IFFT

    xi = np.fft.ifft(x_phase).real
    # --- Overlap and add ---------------
    xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
    x_old = xi[0 + len1:len_]
    k = k + len2
# 保存文件
wf = wave.open('parrots/data/en_output.wav', 'wb')
# 设置参数
wf.setparams(params)
# 设置波形文件 .tostring()将array转换为data
wave_data = (winGain * xfinal).astype(np.short)
wf.writeframes(wave_data.tostring())
wf.close()
"""
"""
"""
"""
    频域滤波降噪 使用傅里叶变换 滤除声音中的噪声
"""
"""
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

# 读取音频文件
sample_rate, noised_signs = wf.read("parrots/data/noised.wav")
print(sample_rate, noised_signs.shape)  # 采样率 (每秒个数), 采样位移
noised_signs = noised_signs / (2 ** 15)
times = np.arange(noised_signs.size) / sample_rate  # x轴

# 绘制音频 时域图
mp.figure("Filter", facecolor="lightgray")
mp.subplot(221)
mp.title("Time Domain", fontsize=12)
mp.ylabel("Noised_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], noised_signs[:200], color="b", label="Noised")
mp.legend()
mp.tight_layout()
mp.show()

# 傅里叶变换 频域分析 音频数据
complex_ary = nf.fft(noised_signs)

fft_freqs = nf.fftfreq(noised_signs.size, times[1] - times[0])  # 频域序列
fft_pows = np.abs(complex_ary)     # 复数的摸-->能量  Y轴

# 绘制频域图
mp.subplot(222)
mp.title("Frequency", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.semilogy(fft_freqs[fft_freqs > 0], fft_pows[fft_freqs > 0], color="orangered", label="Noised")
mp.legend()
mp.tight_layout()

# 去除噪声
fund_freq = fft_freqs[fft_pows.argmax()]
noised_indices = np.where(fft_freqs != fund_freq)
filter_fft = complex_ary.copy()
filter_fft[noised_indices] = 0  # 噪声数据位置 =0
filter_pow = np.abs(filter_fft)

# 绘制去除噪声后的 频域图
mp.subplot(224)
mp.title("Filter Frequency ", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.plot(fft_freqs[fft_freqs > 0], filter_pow[fft_freqs > 0], color="orangered", label="Filter")
mp.legend()
mp.tight_layout()

# 对滤波后的数组，逆向傅里叶变换
filter_sign = nf.ifft(filter_pow).real

# 绘制去除噪声的 时域图像
mp.subplot(223)
mp.title("Filter Time Domain", fontsize=12)
mp.ylabel("filter_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], filter_sign[:200], color="b", label="Filter")
mp.legend()
mp.tight_layout()

# 重新写入新的音频文件
wf.write('parrots/data/0002.wav', sample_rate, (filter_sign * 2 ** 15).astype(np.int16))
"""
import scipy.io.wavfile as wf
from spleeter.separator import Separator
#
# input_ = 'parrots/data/0002.wav'
#
# separator = Separator('spleeter:2stems', multiprocess=False)
# prediction = separator.separate(wf.read(input_)[1])
#
# print(prediction)
# Python -m spleeter separate -i E:/movies/1s.mp3 -p spleeter:2stems -o E:/movies/output
import os
cmd = 'spleeter separate ' + r"G:\speech\parrots-master-new\parrots-master\parrots-master\parrots\data\0032.wav" + ' -p spleeter:2stems -o ' + \
    r"G:\speech\parrots-master-new\parrots-master\parrots-master\parrots\data"
# os.system(cmd)
from glob import glob
from tqdm import tqdm

#1、将所有音频中的人声分离出来
def FetchAudios():
      """
      fetch audios from videos using ffmpeg toolkits
      """
      print("Start Fetch Audios...")
      audio_pathes = sorted(glob(os.path.join(r"G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Processed\audio", '*\*.wav')))
      for audio_path in tqdm(audio_pathes):
            output_path = r"G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Processed\audio" +"\\"+audio_path.split("\\")[-2]
            # output_path = video_path.replace("G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Raw",r"F:\videodata\Processed\audio" ).replace('.mp4', '.wav')
            if not os.path.exists(os.path.dirname(output_path)):
                  os.makedirs(os.path.dirname(output_path))
            # 调用ffmpeg执行音频提取功能
            # cmd = 'ffmpeg -i ' + audio_path + ' -f wav -vn ' + \
            #       output_path + ' -loglevel quiet'
            cmd = 'spleeter separate ' + audio_path + ' -p spleeter:2stems -o ' + output_path
            os.system(cmd)
# FetchAudios()


#2、列出所有的音频
audio_pathes = sorted(glob(os.path.join(r"G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Processed\audio", r'*\*\vocals.wav')))
for audio_path in tqdm(audio_pathes):
      print(audio_path)


#语音数据划分

import shutil,os
wav_path=r"G:\BaiduNetdiskDownload\CH-SIMS\CH-SIMS\Processed\audio\video_0060"
count = 0
wav_files=[]
i=2239

# print('%04d' % (int(36)))

for (dirpath, dirnames, filenames) in os.walk(wav_path):
    # print(dirpath)
    # print(dirnames)
    for filename in filenames:
        #重命名
        if(os.path.basename(filename).split('.')[0]=="vocals"):
            print(filename)
            path=dirpath+r"\vocals.wav"
            # print(path)
            # wav_files.append()
            count+=1
            index = '%04d' % (int(i))
            i += 1
            path1=path.replace("vocals",str(index))
            os.rename(path, path1)
            print(path1)
        #拷贝
        k='%04d' % (int(i))
        # print(k)
        if(os.path.basename(filename).split('.')[0]==str(k)):
            path2=dirpath+"\\"+str(k)+".wav"
            print(path2)
            shutil.copy(path2,r"H:\语音\audio\video_0060")
            i += 1


