# %%
from pyedflib import EdfReader
from utils import edf as myedf,edflab as myedflab
from scipy import signal
DATASET_DIR_PATH = "./dataset/test5"
csv_path = f"{DATASET_DIR_PATH}/gg_export_2022_12_16_170626.csv"
edf_path = f"{DATASET_DIR_PATH}/gg_wasd_test1.edf"
with EdfReader(edf_path) as er:
    fs = int(myedf.get_fs(er))

# %%
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
myedflab.merge_unity2edf(edf_path,csv_path,f"{DATASET_DIR_PATH}/ex.edf",end_marker_offset=1,preprocessing_func=lambda signals:[butter_lowpass_filter(signal,30) for signal in signals])
# %%
