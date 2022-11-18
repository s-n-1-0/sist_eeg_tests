# %%
import pyedflib
from scipy import signal
import utils as myedf
from labedf import csv2,edf2
EDF_PATH = "edf_files/test4/test4_3_0930.edf"

# %% lowpass
# 引用 : https://mori-memo.hateblo.jp/entry/2022/04/30/235815
def butter_lowpass(lowcut, fs, order=4):
    '''バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a
def butter_lowpass_filter(x, lowcut, fs, order=4):
    '''データにローパスフィルタをかける関数
    '''
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y
with pyedflib.EdfReader(EDF_PATH) as redf:
    fs = myedf.get_fs(redf)
    ch = myedf.get_channel_length(redf)
    with pyedflib.EdfWriter("./copy.edf",ch) as wedf:
        header = redf.getHeader()
        header["birthdate"] = ""
        annos = redf.readAnnotations()
        wedf.setHeader(header)
        wedf.setSignalHeaders(redf.getSignalHeaders())
        signals = myedf.get_all_signals(redf)
        wedf.writeSamples([butter_lowpass_filter(s,40,fs) for s in signals])
        for i,_ in enumerate(annos[0]):
            wedf.writeAnnotation(annos[0][i],annos[1][i],annos[2][i])
        wedf.close()
# %% merge
merged_edf_path = "./copy2.edf"
csv2.merge_csv2edf("./copy.edf","./tests/test4/test4--2022-09-30--16_41_53.csv",merged_edf_path)
edf2.split_annotations_edf2hdf(merged_edf_path,"ex.h5",is_groupby=True)
# %%
