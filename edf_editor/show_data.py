#%%
import pyedflib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from tabulate import tabulate
EDF_PATH = "../edf_files/test2_0412_1.edf"
CH_IDX = 1

edf = pyedflib.EdfReader(EDF_PATH)
ewavs = []
for idx,label in enumerate(edf.getSignalLabels()):
    ewavs.append(edf.readSignal(idx))
# %% edfプロパティを表示
headers = ["Prop", "Value"]
signalHeader = edf.getSignalHeader(0)
props = {
    "Ch Size":f"{len(edf.getSignalLabels())}ch",
    "Data Size":edf.getNSamples()[0],
    "Sample Rate":f"{signalHeader['sample_rate']}hz",
    "time [DataSize/SampleRate]":f"{edf.getFileDuration()}s"
}
print(tabulate(props.items(),headers,tablefmt="grid",colalign=('center','center')))
# %% edf波形データを表示
labels = edf.getSignalLabels()
signalHeader = edf.getSignalHeader(0)
times = np.arange(0,edf.getNSamples()[0]) / signalHeader['sample_rate']
plt.figure(figsize=(7,len(labels)*2))
for idx,label in enumerate(labels):
    plt.subplot(len(labels),1,idx + 1)
    plt.plot(times,ewavs[idx],label=labels[idx])
plt.xlabel("Time(s)")
plt.show()
# %% 単純なスペクトログラム
labels = edf.getSignalLabels()
plt.title(f"{labels[CH_IDX]}ch")
spec,freqs,t,*_ = plt.specgram(ewavs[CH_IDX], Fs=250, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default',  scale_by_freq=None, mode='default', scale='default')
spec = 10 * np.log10(spec)
#30hz>フィルター
newFreqsLastIdx = np.where(freqs<=30)[-1][-1]
newFreqs = freqs[:newFreqsLastIdx + 1]
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
plt.pcolormesh(t,newFreqs,spec[:newFreqsLastIdx + 1,:], shading='auto')
plt.show()
# %%
