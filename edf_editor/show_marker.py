#%%
import math
import pyedflib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from tabulate import tabulate
EDF_PATH = "../edf_files/test3_0419.edf"
CH_IDX = 0

edf = pyedflib.EdfReader(EDF_PATH)
ewavs = []
for idx,label in enumerate(edf.getSignalLabels()):
    ewavs.append(edf.readSignal(idx))
labels = edf.getSignalLabels()
annotaitons = edf.readAnnotations()
# %%
signal_header = edf.getSignalHeader(0)
ewav_time = edf.getFileDuration()
rows = []
fs = signal_header['sample_rate']
for ai in range(len(annotaitons[0])):
    name = annotaitons[2][ai]
    time = annotaitons[0][ai]
    idx = math.floor(annotaitons[0][ai] * fs)
    rows.append((name,time,idx))
print(tabulate(rows,["Name","Time(s)","Index"],tablefmt="grid",colalign=('center','center')))
# %%
