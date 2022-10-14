# %%
import pyedflib
from scipy import signal
import edf_viewer as myedf
EDF_PATH = "edf_files/test4/test4_3_0930_run.edf"

# %%
with pyedflib.EdfReader(EDF_PATH) as redf:
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
# %%
