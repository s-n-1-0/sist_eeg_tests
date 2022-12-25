from scipy import signal
# 引用(一部改変) : https://mori-memo.hateblo.jp/entry/2022/04/30/235815

def lowpass_filter(fs,x, lowcut,order=4):
    def make_lowpass(lowcut, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = signal.butter(order, low, btype='low')
        return b, a
    '''データにローパスフィルタをかける関数
    '''
    b, a = make_lowpass(lowcut,order)
    y = signal.filtfilt(b, a, x)
    return y