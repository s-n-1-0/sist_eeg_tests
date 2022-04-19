from scipy import signal
import numpy as np
def get_spectrogram(wav,fs):
    """
    指定した波形のスペクトログラムを求めます。(dB値ではありません)
    """
    result = np.abs(signal.stft(wav,fs=fs, detrend=False, window='hanning', noverlap=128)[2])
    return result
def get_spectrograms(wavs:list,fs:int):
    """
    指定した複数波形をまとめてスペクトログラムに変換します。(dB値ではありません)
    """
    
    def map_get_spectrogram(wav):
        return get_spectrogram(wav,fs)
    result = map(map_get_spectrogram,wavs)
    return np.array(list(result))
