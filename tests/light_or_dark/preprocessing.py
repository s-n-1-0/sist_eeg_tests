# %%
import os
from labedf import csv2,edf2
import numpy as np
from scipy import signal
from pyedflib import EdfReader
from utils.spec import instfreq as _instfreq
from utils import edf as myedf
PROJECT_DATA_DIR_PATH = "./edf_files/lord"
build_dir_path = f"{PROJECT_DATA_DIR_PATH}/build"
with EdfReader(build_dir_path + "/lord_0001.edf") as er:
    fs = int(myedf.get_fs(er))
if not os.path.exists(build_dir_path):
    os.makedirs(build_dir_path)
file_names = [fp.split(".")[0] for fp in os.listdir(f"{PROJECT_DATA_DIR_PATH}/edf")]

# %% merge csv,edf
for file_name in file_names:
    edf_path = f"{PROJECT_DATA_DIR_PATH}/edf/{file_name}.edf"
    csv_path = f"{PROJECT_DATA_DIR_PATH}/csv/{file_name}.csv"
    csv2.merge_csv2edf(edf_path,csv_path,f"{build_dir_path}/{file_name}.edf",label_header_name="LorD")
# %% to hdf
export_path = f"{PROJECT_DATA_DIR_PATH}/ex.h5"
# 引用 : https://mori-memo.hateblo.jp/entry/2022/04/30/235815
def butter_lowpass(lowcut, order=4):
    '''バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a
def butter_lowpass_filter(x, lowcut,order=4):
    '''データにローパスフィルタをかける関数
    '''
    b, a = butter_lowpass(lowcut,order=order)
    y = signal.filtfilt(b, a, x)
    return y
# ---
def norm(s:np.ndarray)->np.ndarray:
    m = np.mean(s,axis=0)
    std = np.std(s,axis=0)
    return (s - m ) /std
def before_preprocessing(signals:list[np.ndarray]):
    signals = [norm(butter_lowpass_filter(signal,200)) for signal in signals] #note take1,2 40 hz -> 3... 200
    return signals
# take1_after_preprocessing = None
def take2_after_preprocessing(signal:np.ndarray):
    """
    瞬時周波数の適用
    """
    def instfreq(x:np.ndarray):
        return _instfreq(x=x,fs=fs,window="hann",nperseg=256,noverlap=250)[0]
    if signal.shape[1] == 0:
        return signal
    first = instfreq(signal[0,:])
    new_signal = np.zeros((signal.shape[0],len(first)))
    new_signal[0,:] = first
    for i in range(signal.shape[0]):
        if i == 0:
            continue
        new_signal[i,:] = instfreq(x=signal[i,:])
    return new_signal
for i ,file_name in enumerate(file_names):
    edf2.split_annotations_edf2hdf(f"{build_dir_path}/{file_name}.edf",
    export_path,
    is_groupby=True,
    is_overwrite= i != 0,
    before_preprocessing_func= before_preprocessing,
    after_preprocessing_func=None
    )

# %% note

with EdfReader(build_dir_path + "/lord_0001.edf") as er:
    s = myedf.get_all_signals(er)[0]
    z = _instfreq(x=norm(butter_lowpass_filter(s,40)),fs=fs,window="hann",nperseg=512)[0]

# %%
