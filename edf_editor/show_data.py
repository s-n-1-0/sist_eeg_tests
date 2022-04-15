#%%
import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tabulate import tabulate
EDF_PATH = "../edf_files/test2_0412_1.edf"
CH_IDX = 0

edf = pyedflib.EdfReader(EDF_PATH)
ewavs = []
for idx,label in enumerate(edf.getSignalLabels()):
    ewavs.append(edf.readSignal(idx))
def getSpecgram(wav):
    return np.abs(signal.stft(wav,fs=250, detrend=False, window='hanning', noverlap=128)[2])
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
logEspecs = 10 * np.log10(especs)
#30hz>フィルター
cutf = np.where((freqs >= 0) & (freqs<=30))
newFreqsLastIdx = cutf[-1][-1]
newFreqsFirstIdx = cutf[0][0]
newFreqs = freqs[newFreqsFirstIdx:newFreqsLastIdx + 1]

plt.figure()
plt.title("avg ch")
plt.pcolormesh(t,freqs,np.mean(logEspecs,axis=0), shading='auto')
plt.show()
plt.title(f"avg ch")
plt.pcolormesh(t,newFreqs,np.mean(logEspecs[:,newFreqsFirstIdx:newFreqsLastIdx + 1,:],axis=0), shading='auto')
plt.show()
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
_ = plt.specgram(ewavs[CH_IDX], Fs=250, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default',  scale_by_freq=None, mode='default', scale='default')
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
plt.pcolormesh(t,newFreqs,logEspecs[CH_IDX,newFreqsFirstIdx:newFreqsLastIdx + 1,:], shading='auto')
plt.show()
# %%  α波、β波グラフをプロット
cutt = np.where((t>=0) & (t<60)) #指定した範囲で集計
newTime = t[cutt[0][0]:cutt[-1][-1]+1]
cutfAlpha = np.where((freqs >= 8) & (freqs<=13))
cutfBeta = np.where((freqs > 13) & (freqs<=25))
alphaFreqs = freqs[cutfAlpha[0][0]:cutfAlpha[-1][-1] + 1]
betaFreqs = freqs[cutfBeta[0][0]:cutfBeta[-1][-1] + 1]
print(especs.shape)
alphaPower = np.mean(especs[0,cutfAlpha[0][0]:cutfAlpha[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
betaPower = np.mean(especs[0,cutfBeta[0][0]:cutfBeta[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
#alphaAvgPower = np.mean(especs[:,cutfAlpha[0][0]:cutfAlpha[-1][-1] + 1,:],axis=(0,1))
#betaAvgPower = np.mean(especs[:,cutfBeta[0][0]:cutfBeta[-1][-1] + 1,:],axis=(0,1))
plt.plot(newTime,alphaPower,label="α")
plt.plot(newTime,betaPower,label="β")
plt.plot(newTime, betaPower / alphaPower,label="β/α")
plt.xlabel("Time(s)")
plt.legend()
#plt.ylim(10,-10)
plt.show()
print(f"α:{np.mean(alphaPower)}")
print(f"β:{np.mean(betaPower)}")
print(f"β/:{np.mean(betaPower/alphaPower)}")

# %%
