from typing import Any
from scipy.signal import stft
import numpy as np

def get_spectrogram(wav,fs,nperseg=None,noverlap=128):
    """
    指定した波形のスペクトログラムを求めます。(dB値ではありません)
    """
    f,t,z = stft(wav,fs=fs,nperseg=nperseg,detrend=False, window='hanning', noverlap=noverlap)
    return (f,t,np.abs(z))
def get_spectrograms(wavs:list,fs:int,nperseg=None,noverlap=128,conv_array=True):
    """
    指定した複数波形をまとめてスペクトログラムに変換します。(dB値ではありません)
    ※この関数に入力される全ての波形のサイズは同じサイズである必要があります。
    """
    csize = len(wavs[0])
    len_wavs = [len(wav) for wav in wavs]
    assert all([lw == csize for lw in len_wavs])
    def map_get_spectrogram(wav):
        return get_spectrogram(wav,fs,nperseg=nperseg,noverlap=noverlap)
    result = list(map(map_get_spectrogram,wavs))
    if conv_array:
        return (result[0][0],result[0][1], np.array([r[2] for r in result]))
    else:
        return (result[0][0],result,[0][1], [r[2] for r in result])

def instfreq(**stft_args):
    """
    瞬時周波数を求めます。notes/instfreqを参照。
    引数はscipy.signal.stftの引数と同じです(ただし全ての引数はstftの引数名を指定する必要があります)
    Returns (moment,times)
    """

    freq,times,spec = stft(**stft_args)
    power = np.abs(spec) ** 2
    tfd = power
    tfd = tfd / np.sum(tfd)
    tfdSum = np.sum(tfd,axis=0)
    for i in range(tfd.shape[1]):
        tfd[:,i] *= freq
    tmp = np.sum(tfd,axis=0)
    moment = tmp / tfdSum
    return moment,times
