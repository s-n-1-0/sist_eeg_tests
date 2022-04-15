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
def get_specgram(wav):
    return np.abs(signal.stft(wav,fs=250, detrend=False, window='hanning', noverlap=128)[2])
freqs,t,_ = signal.stft(ewavs[0],fs=250, detrend=False, window='hanning', noverlap=128)
especs = np.array(list(map(get_specgram,ewavs)))
labels = edf.getSignalLabels()
# %% edfプロパティを表示
headers = ["Prop", "Value"]
signal_header = edf.getSignalHeader(0)
props = {
    "Ch Size":f"{len(labels)}ch",
    "Data Size":edf.getNSamples()[0],
    "Sample Rate":f"{signal_header['sample_rate']}hz",
    "time [DataSize/SampleRate]":f"{edf.getFileDuration()}s"
}
print(tabulate(props.items(),headers,tablefmt="grid",colalign=('center','center')))
# %% edf波形データを表示
signal_header = edf.getSignalHeader(0)
times = np.arange(0,edf.getNSamples()[0]) / signal_header['sample_rate']
plt.figure(figsize=(7,len(labels)*2))
for idx,label in enumerate(labels):
    plt.subplot(len(labels),1,idx + 1)
    plt.plot(times,ewavs[idx],label=labels[idx])
plt.xlabel("Time(s)")
plt.show()
# %% 単純なスペクトログラム
log_especs = 10 * np.log10(especs)
#30hz>フィルター
cutf = np.where((freqs >= 0) & (freqs<=30))
new_freqs_last_idx = cutf[-1][-1]
new_freqs_first_idx = cutf[0][0]
new_freqs = freqs[new_freqs_first_idx:new_freqs_last_idx + 1]

plt.figure()
plt.title("avg ch")
plt.pcolormesh(t,freqs,np.mean(log_especs,axis=0), shading='auto')
plt.show()
plt.title(f"avg ch")
plt.pcolormesh(t,new_freqs,np.mean(log_especs[:,new_freqs_first_idx:new_freqs_last_idx + 1,:],axis=0), shading='auto')
plt.show()
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
_ = plt.specgram(ewavs[CH_IDX], Fs=250, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default',  scale_by_freq=None, mode='default', scale='default')
plt.figure()
plt.title(f"{labels[CH_IDX]}ch")
plt.pcolormesh(t,new_freqs,log_especs[CH_IDX,new_freqs_first_idx:new_freqs_last_idx + 1,:], shading='auto')
plt.show()
# %%  α波、β波グラフをプロット
cutt = np.where((t>=0) & (t<60)) #指定した範囲で集計
new_time = t[cutt[0][0]:cutt[-1][-1]+1]
cutf_alpha = np.where((freqs >= 8) & (freqs<=13))
cutf_beta = np.where((freqs > 13) & (freqs<=25))
alpha_freqs = freqs[cutf_alpha[0][0]:cutf_alpha[-1][-1] + 1]
beta_feqs = freqs[cutf_beta[0][0]:cutf_beta[-1][-1] + 1]
print(especs.shape)
alpha_power = np.mean(especs[0,cutf_alpha[0][0]:cutf_alpha[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
beta_power = np.mean(especs[0,cutf_beta[0][0]:cutf_beta[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
#alphaAvgPower = np.mean(especs[:,cutf_alpha[0][0]:cutf_alpha[-1][-1] + 1,:],axis=(0,1))
#betaAvgPower = np.mean(especs[:,cutf_beta[0][0]:cutf_beta[-1][-1] + 1,:],axis=(0,1))
plt.plot(new_time,alpha_power,label="α")
plt.plot(new_time,beta_power,label="β")
plt.plot(new_time, beta_power / alpha_power,label="β/α")
plt.xlabel("Time(s)")
plt.legend()
#plt.ylim(10,-10)
plt.show()
print(f"α:{np.mean(alpha_power)}")
print(f"β:{np.mean(beta_power)}")
print(f"β/:{np.mean(beta_power/alpha_power)}")

# %%
