# %%
import os
from labedf import csv2,edf2
import numpy as np
from scipy import signal
from pyedflib import EdfReader
from utils.spec import instfreq as _instfreq
from utils import edf as myedf,edflab as myedflab,signals_standardization
PROJECT_DATA_DIR_PATH = "./dataset/lord2"
build_dir_path = f"{PROJECT_DATA_DIR_PATH}/build"
edfcsv_filenames = myedflab.get_edfcsv_filenames(f"{PROJECT_DATA_DIR_PATH}/ペア.csv")
with EdfReader(f"{PROJECT_DATA_DIR_PATH}/edf/{edfcsv_filenames[0,0]}") as er:
    fs = int(myedf.get_fs(er))
if not os.path.exists(build_dir_path):
    os.makedirs(build_dir_path)

# %% merge csv,edf
filenames = []
for i in range(edfcsv_filenames.shape[0]):
    edf_path = f"{PROJECT_DATA_DIR_PATH}/edf/{edfcsv_filenames[i,0]}"
    csv_path = f"{PROJECT_DATA_DIR_PATH}/csv/{edfcsv_filenames[i,1]}"
    filename = f"merged_{i}"
    csv2.merge_csv2edf(edf_path,csv_path,f"{build_dir_path}/{filename}.edf",label_header_name="LorD")
    filenames.append(filename)
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
def before_preprocessing(signals:list[np.ndarray]):
    signals = [butter_lowpass_filter(signal,30) for signal in signals]
    return signals
def after_preprocessing(signals:np.ndarray,label:str):
    if label != "dark" and label != "light":
        return signals
    return signals_standardization(signals)
for i ,filename in enumerate(filenames):
    edf2.split_annotations_edf2hdf(f"{build_dir_path}/{filename}.edf",
    export_path,
    is_groupby=True,
    is_overwrite= i != 0,
    before_preprocessing_func= before_preprocessing,
    after_preprocessing_func=after_preprocessing
    )

# %%
