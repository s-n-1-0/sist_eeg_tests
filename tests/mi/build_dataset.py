#
# 必要なセルだけ実行すること。全て実行してもいいけどかなり時間がかかる
#

#%%
import numpy as np
import h5py
from scipy import signal
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import glob
import os
import mne
from bci_dataset import DatasetUpdater
import pywt
from scipy.signal import butter, filtfilt
root_path = "C:/Users/2221012/OneDrive - 静岡理工科大学"#//172.16.88.200/private/2221012"
p3_dir = "MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms"
mi52_dir = "MI_100295/mat_data"

p3_h5_name = "3pdataset_fix.h5"
mla_h5_name = "mla.h5"
merged_h5_name = "merged_mla2.h5"
fs = 500
ch = 14 #C4


#%%
p3_info_list = []
mi52_info_list = []
file_paths = glob.glob(root_path + p3_dir + "/pres_rest/*.set")
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    fns = file_name.split("_")
    session = int(fns[0][4:])
    subject = int(fns[1][4:])
    p3_info_list.append((file_path,session,subject))
file_paths = glob.glob(root_path + mi52_dir + "/pres/*.set")
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    subject,label = file_name.split(".")[0].split("_")
    subject = int(subject[1:])
    mi52_info_list.append((file_path,1,subject,label))
print(len(p3_info_list),len(mi52_info_list))
#%% Preview
preview_path,session,subject, = p3_info_list[6]
x = mne.io.read_epochs_eeglab(preview_path).get_data(item=["left","right"])
x.shape,session,subject
# 標準化
dx = x[0,:,:]
dx = StandardScaler().fit_transform(dx.T).T
#test: np.array([[1,2,3],[10,20,30]]),StandardScaler().fit_transform(np.array([[1,2,3],[10,20,30]]).T).T

# パワースペクトルに変換
freq, px = signal.periodogram(dx[:,500:1000], fs=fs)
trimmed_freq_flags = (freq >= 8) & (freq <= 20)
print(len(freq[trimmed_freq_flags]),np.where(trimmed_freq_flags)[0][0],np.where(trimmed_freq_flags)[0][-1])
plt.plot(freq[trimmed_freq_flags],np.mean(px,axis=0)[trimmed_freq_flags])
plt.ylim((0,0.050))

# DWT
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

# %% 前処理関数の定義
def bandpass(data):
    lowcut = 8  # バンドパスフィルタの下限周波数
    highcut = 30  # バンドパスフィルタの上限周波数
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    data = filtfilt(b, a, data)
    return data

# %% 3p定義
p3updater = DatasetUpdater(hdf_path=root_path+"/"+p3_h5_name,
                        fs=fs,dataset_name="3p")
# %% 3p初期化＆追加
p3updater.remove_hdf()
for path,session,subject in p3_info_list:
    p3updater.add_eeglab(path,
                        ["left","right"],
                        {"session":int(session),"subject":int(subject)})

# %% mla定義(MLA.h5は別リポジトリで既に構築済み)
mlaupdater = DatasetUpdater(hdf_path=root_path+"/"+mla_h5_name,
                        fs=fs,
                        dataset_name="mla")

# %%
updater = DatasetUpdater(hdf_path="C:/Users/2221012/OneDrive - 静岡理工科大学/"+merged_h5_name,
                        fs=fs)
updater.remove_hdf()
updater.merge_hdf(p3updater,ch_indexes=[7, 8, 9, 10, 12, 35, 13, 36, 14, 17, 18, 19, 20])
updater.merge_hdf(mlaupdater,ch_indexes=list(range(13)))
#std
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    return StandardScaler().fit_transform(x.T).T #標準化
updater.preprocess("std",prepro_func)

# %% psd(必要だったら実行)
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    x = StandardScaler().fit_transform(x.T).T
    _, px = signal.periodogram(x[:,500:1000], fs=fs)
    return px
updater.preprocess("psd",prepro_func)


# %% DWT prepro(必要だったら実行)
import pywt
def dwt(ch_signal):
    wavelet = 'db4'  # Daubechies 4 wavelet
    level = pywt.dwt_max_level(len(ch_signal), wavelet)  # maximum feasible level
    coefficients = pywt.wavedec(ch_signal, wavelet, level=level)
    return coefficients[4:6]
def prepro_func(x:np.ndarray):
    x = bandpass(x)
    x = StandardScaler().fit_transform(x.T).T
    w = []
    for i in range(x.shape[0]):
        ch_dx = x[i,500:1000]
        _w = dwt(ch_dx)
        _w = [np.concatenate([__w, np.zeros(len(ch_dx) - len(__w))]) for __w in _w]
        w.append(_w)
    wx = np.array(w)
    return wx
updater.preprocess("dwt",prepro_func)
