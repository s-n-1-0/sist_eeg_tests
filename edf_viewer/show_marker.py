#%%
import lib.spec as spec
import lib.edf as myedf
import math
import pyedflib
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from tabulate import tabulate
EDF_PATH = "../edf_files/test3_0419.edf"
CH_IDX = 0

edf = pyedflib.EdfReader(EDF_PATH)
ewavs = []
for idx,label in enumerate(edf.getSignalLabels()):
    ewavs.append(edf.readSignal(idx))
labels = edf.getSignalLabels()
annotaitons = edf.readAnnotations()
fs = myedf.get_fs(edf)
# %%
signal_header = edf.getSignalHeader(0)
ewav_time = edf.getFileDuration()
rows = []
fs = signal_header['sample_rate']
for ai in range(len(annotaitons[0])):
    name = annotaitons[2][ai]
    time = annotaitons[0][ai]
    idx = math.floor(annotaitons[0][ai] * fs)
    rows.append((name,time,idx))
print(tabulate(rows,["Name","Time(s)","Index"],tablefmt="grid",colalign=('center','center')))
# %% 【前のセル実行必須】平均波形にアノテーションを表示します。
freqs,t,_ = signal.stft(ewavs[0],fs=fs, detrend=False, window='hanning', noverlap=128)
especs = spec.get_spectrograms(ewavs,fs)
log_especs = 10 * np.log10(especs)
plt.figure()
plt.title("avg ch")
plt.pcolormesh(t,freqs,np.mean(log_especs,axis=0), shading='auto')
for _,r,_ in rows:
    plt.vlines(r, 0, freqs[-1],colors='#FF4C4C',linestyle='dashed', linewidth=3)
plt.show()
plt.show()
# %%
