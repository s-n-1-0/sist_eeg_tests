# %%
import numpy as np
def signals_standardization(s:np.ndarray):
    """
    信号ごとに標準化します。
    """
    if len(s.shape) == 1:
        return std(s)
    m = np.mean(s,axis=1)
    _std = np.std(s,axis=1)
    for i in range(m.shape[0]):
        s[i,:] = (s[i,:] - m[i]) / _std[i]        
    return s
def standardization(s:np.ndarray):
    """
    指定したデータ全体を標準化します。
    """
    m = np.mean(s,axis=0)
    _std = np.std(s,axis=0)
    return (s - m ) /_std
