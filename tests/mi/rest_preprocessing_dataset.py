# %%
import numpy as np
import scipy
import h5py
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
root_path = "//172.16.88.200/private/2221012/"
p3_dir = "MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms"
mi52_dir = "MI_100295/mat_data"
fs = 500
ch = 14 #C4
#%% preview
def downsample_multichannel_signal(input_signal, original_rate, target_rate):
    num_channels = input_signal.shape[0]  # チャネル数
    num_samples = input_signal.shape[1]  # サンプル数
    # ダウンサンプリング後のサンプル数を計算
    target_samples = int(num_samples * (target_rate / original_rate))
    # 出力信号の配列を作成
    downsampled_signal = np.zeros((num_channels, target_samples))
    for channel in range(num_channels):
        # ダウンサンプリングするチャネルの信号を取得
        channel_signal = input_signal[channel, :]
        # ダウンサンプリング
        downsampled_channel_signal = signal.resample(channel_signal, target_samples)
        # 出力信号に格納
        downsampled_signal[channel, :] = downsampled_channel_signal
    return downsampled_signal
train_key = "EEG_MI_train"
def p3_parse_mat(parent_key,mat_data):
    data = mat_data[parent_key][0][0]
    #形式整理
    for i in range(len(data)):
        sq_data = data[i].squeeze()
        if len(sq_data.shape) == 1:
            data[i] =  sq_data
        if len(sq_data.shape) == 0:
            data[i] = sq_data
    return data
def convert_average_spectrum(data):
    frequencies, times, Sxx = spectrogram(data, fs, nperseg=1024)
    # 指定した周波数帯（8Hzから30Hz）にフィルタリング
    low_freq = 8
    high_freq = 30
    mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    filtered_Sxx = Sxx[mask, :]
    # 時間方向に平均
    average_spectrum = np.mean(filtered_Sxx, axis=1)
    return average_spectrum
session = scipy.io.loadmat(f"{root_path+mi52_dir}/s{(1):02}.mat")
train_rest = downsample_multichannel_signal(session["eeg"][0][0][1],512,500)
train_rest =  StandardScaler().fit_transform(train_rest.T).T
train_rest = train_rest[[i - 1 for i in [9,11,46,44,13,12,48,49,50,17,19,56,54]],:]
_train_rest = []
for k in range(train_rest.shape[0]):
    _train_rest.append(convert_average_spectrum(train_rest[k,:]))
np.array(_train_rest).shape
#%% restデータ統合
with h5py.File(root_path+"/rest.h5",mode="w") as h5:
    if "rest" in h5:
        del h5["rest"]
    rest_group = h5.require_group("rest")
    for i, size in enumerate((54,54)):
        for j in range(size):
            session = scipy.io.loadmat(f"{root_path+p3_dir}/session{i+1}/sess{(i+1):02}_subj{(j+1):02}_EEG_MI.mat")
            train_rest = downsample_multichannel_signal(p3_parse_mat(train_key,session)[13].T,1000,500)
            train_rest =  StandardScaler().fit_transform(train_rest.T).T
            train_rest = train_rest[[7, 8, 9, 10, 12, 35, 13, 36, 14, 17, 18, 19, 20],:]
            _train_rest = []
            for k in range(train_rest.shape[0]):
                _train_rest.append(convert_average_spectrum(train_rest[k,:]))
            train_rest = np.array(_train_rest)
            session_group = rest_group.require_group("3p").require_group(str(i+1))
            session_group.create_dataset(str(j+1),train_rest.shape,data=train_rest)
    i = 0
    size = 52
    rest_group = h5.require_group("rest")
    for j in range(size):
        session = scipy.io.loadmat(f"{root_path+mi52_dir}/s{(i+1):02}.mat")
        train_rest = downsample_multichannel_signal(session["eeg"][0][0][1],512,500)
        train_rest =  StandardScaler().fit_transform(train_rest.T).T
        train_rest = train_rest[[i - 1 for i in [9,11,46,44,13,12,48,49,50,17,19,56,54]],:]
        _train_rest = []
        for k in range(train_rest.shape[0]):
            _train_rest.append(convert_average_spectrum(train_rest[k,:]))
        train_rest = np.array(_train_rest)
        session_group = rest_group.require_group("mi52").require_group(str(i+1))
        session_group.create_dataset(str(j+1),train_rest.shape,data=train_rest)



# %%
