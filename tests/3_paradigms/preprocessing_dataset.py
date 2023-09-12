#%%
import numpy as np
import h5py
from scipy import signal
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import glob
import os
import mne
from eeghdf import EEGHDFUpdater
root_path = "MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms"
file_paths = glob.glob(root_path + "/pres/*.set")
fs = 500
ch = 14 #C4


#%%
finfo_list = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    fns = file_name.split("_")
    session = int(fns[0][4:])
    subject = int(fns[1][4:])
    finfo_list.append((file_path,session,subject))


# 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'

#%%
eeglist = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
using_lst = ['FC5','FC1','FC2','FC6','C3','C1','Cz','C2','C4','CP5','CP1','CP2','CP6']
#using_lst = ['C3','Cz','C4']
ch_indexes = []
for item in using_lst:
    ch_indexes.append(eeglist.index(item))

len(ch_indexes),ch_indexes


# ## Preview
#%%
preview_path,session,subject, = finfo_list[6]
x = mne.io.read_epochs_eeglab(preview_path).get_data(item=["left","right"])
x.shape,session,subject


# ### 標準化
#%%


dx = x[0,:,:]
dx = StandardScaler().fit_transform(dx.T).T
#test: np.array([[1,2,3],[10,20,30]]),StandardScaler().fit_transform(np.array([[1,2,3],[10,20,30]]).T).T
dx.shape


# ### パワースペクトルに変換
# %%


# パワースペクトルへの変換
freq, px = signal.periodogram(dx[:,500:1000], fs=fs)
trimmed_freq_flags = (freq >= 8) & (freq <= 20)
print(len(freq[trimmed_freq_flags]),np.where(trimmed_freq_flags)[0][0],np.where(trimmed_freq_flags)[0][-1])
plt.plot(freq[trimmed_freq_flags],np.mean(px,axis=0)[trimmed_freq_flags])
plt.ylim((0,0.050))


# ### DWT
# %%
import pywt
def dwt(ch_signal):
    wavelet = 'db4'  # Daubechies 4 wavelet
    level = pywt.dwt_max_level(len(ch_signal), wavelet)  # maximum feasible level
    coefficients = pywt.wavedec(ch_signal, wavelet, level=level)
    return coefficients[4:6]
w = []
for i in range(dx.shape[0]):
    ch_dx = dx[i,500:1000]
    _w = dwt(ch_dx)
    _w = [np.concatenate([__w, np.zeros(len(ch_dx) - len(__w))]) for __w in _w]
    w.append(_w)
w = np.array(w)
print(w.shape)


# ## まとめて処理(trainのみ)
# %%
updater = EEGHDFUpdater(hdf_path=root_path+"/3pdataset.h5",
                        fs=fs,
                        lables=["left","right"])
updater.remove_hdf()
for path,session,subject in finfo_list:
    updater.add_eeglab(path,{"session":int(session),"subject":int(subject)})


# %%
from scipy.signal import butter, filtfilt
def bandpass(data):
    lowcut = 8  # バンドパスフィルタの下限周波数
    highcut = 30  # バンドパスフィルタの上限周波数
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    data = filtfilt(b, a, data)
    return data


# %%
updater = EEGHDFUpdater(hdf_path=root_path+"/3pdataset.h5",
                        fs=fs,
                        lables=["left","right"])
#std
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    return StandardScaler().fit_transform(x.T).T #標準化
updater.preprocess("std",prepro_func)

#psd
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    x = StandardScaler().fit_transform(x.T).T
    _, px = signal.periodogram(x[:,500:1000], fs=fs)
    return px
updater.preprocess("psd",prepro_func)


# %%
import pywt
updater = EEGHDFUpdater(hdf_path=root_path+"/3pdataset.h5",
                        fs=fs,
                        lables=["left","right"])
def dwt(ch_signal):
    wavelet = 'db4'  # Daubechies 4 wavelet
    level = pywt.dwt_max_level(len(ch_signal), wavelet)  # maximum feasible level
    coefficients = pywt.wavedec(ch_signal, wavelet, level=level)
    return coefficients[4:6]
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    x = StandardScaler().fit_transform(x.T).T
    w = []
    for i in range(dx.shape[0]):
        ch_dx = x[i,500:1000]
        _w = dwt(ch_dx)
        _w = [np.concatenate([__w, np.zeros(len(ch_dx) - len(__w))]) for __w in _w]
        w.append(_w)
    wx = np.array(w)
    return wx
updater.preprocess("dwt",prepro_func)


# ## restデータ統合
# %%
import scipy
from scipy import signal
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
dataset_size = (54,54)
train_key = "EEG_MI_train"
def parse_mat(parent_key,mat_data):
    data = mat_data[parent_key][0][0]
    #形式整理
    for i in range(len(data)):
        sq_data = data[i].squeeze()
        if len(sq_data.shape) == 1:
            data[i] =  sq_data
        if len(sq_data.shape) == 0:
            data[i] = sq_data
    return data
with h5py.File(root_path+"/3pdataset.h5",mode="r+") as h5:
    if "rest" in h5:
        del h5["rest"]
    rest_group = h5.require_group("rest")
    for i, size in enumerate(dataset_size):
        for j in range(size):
            session = scipy.io.loadmat(f"{root_path}/session{i+1}/sess{(i+1):02}_subj{(j+1):02}_EEG_MI.mat")
            train_rest = downsample_multichannel_signal(parse_mat(train_key,session)[13].T,1000,500)
            train_rest =  StandardScaler().fit_transform(train_rest.T).T
            session_group = rest_group.require_group(str(i+1))
            session_group.create_dataset(str(j+1),train_rest.shape,data=train_rest)

