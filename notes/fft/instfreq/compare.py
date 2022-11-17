# %%
import numpy as np
from scipy.signal import chirp,stft,spectrogram
from scipy.fft import fft,ifft
import matplotlib.pyplot as plt
fs = 3000
t = np.linspace(0,2,fs*2)
y = chirp(t, f0=100, t1=1,f1=200,  method='linear') #note :  "quadratic"

# %%
spec = fft(y)
spec[3000:] = 0
spec = spec * 2
z = ifft(spec)
re = np.real(z)
im = np.imag(z)
phase = np.arctan2(im,re)
phase = np.unwrap(phase)
ifreq = (fs/(2*np.pi)) * np.gradient(phase)
plt.plot(t,ifreq)
# %%
freq,times,spec = stft(y,fs=fs,window="hann",nperseg=1024*2,noverlap=2017)
power = np.abs(spec) ** 2
tfd = power
tfd = tfd / np.sum(tfd)
tfdSum = np.sum(tfd,axis=0)
for i in range(tfd.shape[1]):
    tfd[:,i] *= freq
tmp = np.sum(tfd,axis=0)
moment = tmp / tfdSum
plt.pcolormesh(times, freq, power)
plt.show()
plt.plot(times,moment)
# %%
