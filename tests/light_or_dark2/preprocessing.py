# %%
import os
from labedf import csv2,edf2
import numpy as np
from scipy import signal
from pyedflib import EdfReader
from utils import edf as myedf,edflab as myedflab,signals_standardization
DATASET_DIR_PATH = "./dataset/lord2/train"
file_settings = myedflab.MergeAllCsv2EdfFileSettings(DATASET_DIR_PATH + "/ペア.csv",list_encoding="ansi")
edfcsv_filenames = file_settings.get_edfcsv_filenames()
with EdfReader(f"{DATASET_DIR_PATH}/edf/{edfcsv_filenames[0,0]}") as er:
    fs = int(myedf.get_fs(er))

# %% merge csv,edf
filenames = myedflab.merge_all_csv2edf(file_settings,label_header_name="LorD")
filenames
# %% to hdf
export_path = f"{DATASET_DIR_PATH}/ex.h5"
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
    edf2.split_annotations_edf2hdf(f"{file_settings.build_dir_path}/{filename}.edf",
    export_path,
    is_groupby=True,
    is_overwrite= i != 0,
    before_preprocessing_func= before_preprocessing,
    after_preprocessing_func=after_preprocessing
    )

# %%
