from scipy import signal
import numpy as np
from sqlalchemy import true
def get_spectrogram(wav,fs,nperseg=None,noverlap=128):
    """
    指定した波形のスペクトログラムを求めます。(dB値ではありません)
    """
    result = np.abs(signal.stft(wav,fs=fs,nperseg=nperseg,detrend=False, window='hanning', noverlap=noverlap)[2])
    return result
def get_spectrograms(wavs:list,fs:int,nperseg=None,noverlap=128,conv_array=True):
    """
    指定した複数波形をまとめてスペクトログラムに変換します。(dB値ではありません)
    """
    
    def map_get_spectrogram(wav):
        return get_spectrogram(wav,fs,nperseg=nperseg,noverlap=noverlap)
    result = map(map_get_spectrogram,wavs)
    if conv_array:
        return np.array(list(result))
    else:
        return list(result)
