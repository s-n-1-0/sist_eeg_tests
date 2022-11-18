# %%
import os
from labedf import csv2,edf2
import numpy as np
from scipy import signal
from utils.spec import instfreq
PROJECT_DATA_DIR_PATH = "./edf_files/lord"
fs = 500  # note: できればファイルから取得するべき
build_dir_path = f"{PROJECT_DATA_DIR_PATH}/build"
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
def take1_preprocessing(signals:list[np.ndarray]):
    signals = [norm(butter_lowpass_filter(signal,40)) for signal in signals]
    return signals
def take2_preprocessing(signals:list[np.ndarray]):
    signals = [norm(butter_lowpass_filter(signal,40)) for signal in signals]
    return signals
for i ,file_name in enumerate(file_names):
    edf2.split_annotations_edf2hdf(f"{build_dir_path}/{file_name}.edf",
    export_path,
    is_groupby=True,
    is_overwrite= i != 0,
    preprocessing_func=take2_preprocessing)

# %% note
from pyedflib import EdfReader
from utils import edf
with EdfReader(build_dir_path + "/lord_0001.edf") as er:
    s = edf.get_all_signals(er)[0]
    z = instfreq(x=norm(butter_lowpass_filter(s,40)),fs=fs,window="hann",nperseg=512)[0]

# %%
