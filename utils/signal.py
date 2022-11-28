import numpy as np
def norm(s:np.ndarray):
    if len(s.shape) == 1:
        m = np.mean(s,axis=0)
        std = np.std(s,axis=0)
        return (s - m ) /std
    m = np.mean(s,axis=1)
    std = np.std(s,axis=1)
    for i in range(m.shape[0]):
        s[i,:] = (s[i,:] - m[i]) / std[i]        
    return s