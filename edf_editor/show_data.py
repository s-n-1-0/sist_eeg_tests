#%%
from pandas import cut
import pyedflib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from tabulate import tabulate
EDF_PATH = "../edf_files/test2_0412_1.edf"
CH_IDX = 1

edf = pyedflib.EdfReader(EDF_PATH)
ewavs = []
for idx,label in enumerate(edf.getSignalLabels()):
    ewavs.append(edf.readSignal(idx))
def getSpecgram(wav):
    return 10 * np.log10(np.abs(signal.stft(wav,fs=250, detrend=False, window='hanning', noverlap=128)[2]))
freqs,t,_ = signal.stft(ewavs[0],fs=250, detrend=False, window='hanning', noverlap=128)
especs = np.array(list(map(getSpecgram,ewavs)))
labels = edf.getSignalLabels()
# %% edfプロパティを表示
headers = ["Prop", "Value"]
signalHeader = edf.getSignalHeader(0)
props = {
    "Ch Size":f"{len(labels)}ch",
    "Data Size":edf.getNSamples()[0],
    "Sample Rate":f"{signalHeader['sample_rate']}hz",
    "time [DataSize/SampleRate]":f"{edf.getFileDuration()}s"
}
print(tabulate(props.items(),headers,tablefmt="grid",colalign=('center','center')))
# %% edf波形データを表示
signalHeader = edf.getSignalHeader(0)
times = np.arange(0,edf.getNSamples()[0]) / signalHeader['sample_rate']
plt.figure(figsize=(7,len(labels)*2))
for idx,label in enumerate(labels):
    plt.subplot(len(labels),1,idx + 1)
    plt.plot(times,ewavs[idx],label=labels[idx])
plt.xlabel("Time(s)")
plt.show()
# %% 単純なスペクトログラム
#30hz>フィルター
cutf = np.where((freqs >= 0) & (freqs<=30))
newFreqsLastIdx = cutf[-1][-1]
newFreqsFirstIdx = cutf[0][0]
newFreqs = freqs[newFreqsFirstIdx:newFreqsLastIdx + 1]

plt.figure()
plt.title("avg ch")
plt.pcolormesh(t,freqs,np.mean(especs,axis=0), shading='auto')
plt.show()
plt.title(f"avg ch")
plt.pcolormesh(t,newFreqs,np.mean(especs[:,newFreqsFirstIdx:newFreqsLastIdx + 1,:],axis=0), shading='auto')
plt.show()
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
_ = plt.specgram(ewavs[CH_IDX], Fs=250, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default',  scale_by_freq=None, mode='default', scale='default')
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
plt.pcolormesh(t,newFreqs,especs[CH_IDX,newFreqsFirstIdx:newFreqsLastIdx + 1,:], shading='auto')
plt.show()
# %%
