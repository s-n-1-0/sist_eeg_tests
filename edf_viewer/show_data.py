#%%
import lib.spec as spec
import lib.edf as myedf
import pyedflib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from tabulate import tabulate
EDF_PATH = "../edf_files/test2_0412_2.edf"
CH_IDX = 0

edf = pyedflib.EdfReader(EDF_PATH)
fs = myedf.get_fs(edf)
ewavs = myedf.get_all_signals(edf)
freqs,t,_ = signal.stft(ewavs[0],fs=fs, detrend=False, window='hanning', noverlap=128)
especs = spec.get_spectrograms(ewavs,fs)
labels = edf.getSignalLabels()
# %% edfプロパティを表示
headers = ["Prop", "Value"]
signal_header = edf.getSignalHeader(0)
props = {
    "Ch Size":f"{len(labels)}ch",
    "Data Size":edf.getNSamples()[0],
    "Sample Rate":f"{fs}hz",
    "time [DataSize/SampleRate]":f"{edf.getFileDuration()}s"
}
print(tabulate(props.items(),headers,tablefmt="grid",colalign=('center','center')))
# %% edf波形データを表示
signal_header = edf.getSignalHeader(0)
times = np.arange(0,edf.getNSamples()[0]) / fs
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
_ = plt.specgram(ewavs[CH_IDX], Fs=fs, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default',  scale_by_freq=None, mode='default', scale='default')
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
alpha_power = np.mean(especs[CH_IDX,cutf_alpha[0][0]:cutf_alpha[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
beta_power = np.mean(especs[CH_IDX,cutf_beta[0][0]:cutf_beta[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=0)
alpha_avg_power = np.mean(especs[:,cutf_alpha[0][0]:cutf_alpha[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=(0,1))
beta_avg_power = np.mean(especs[:,cutf_beta[0][0]:cutf_beta[-1][-1] + 1,cutt[0][0]:cutt[-1][-1]+1],axis=(0,1))
plt.title("avg ch : α,β,β/α")
plt.plot(new_time,alpha_avg_power,label="α")
plt.plot(new_time,beta_avg_power,label="β")
plt.plot(new_time, beta_avg_power / alpha_avg_power,label="β/α")
plt.xlabel("Time(s)")
plt.legend()
#plt.ylim(10,-10)
plt.show()
props = {
    "Avg ch・α":f"{np.mean(alpha_avg_power)}",
    "Avg ch・β":f"{np.mean(beta_avg_power)}",
    "Avg ch・β/α":f"{np.mean(beta_avg_power/alpha_avg_power)}"
}
print("---Avg ch---")
print(tabulate(props.items(),["Type","Value"],tablefmt="grid",colalign=('center','center')))
props = {
    f"{labels[CH_IDX]}ch・α":f"{np.mean(alpha_power)}",
    f"{labels[CH_IDX]}ch・β":f"{np.mean(beta_power)}",
    f"{labels[CH_IDX]}ch・β/α":f"{np.mean(beta_power/alpha_power)}"
}
print("\n\n")
print(f"---{labels[CH_IDX]}ch---")
print(tabulate(props.items(),["Type","Value"],tablefmt="grid",colalign=('center','center')))
# %%
