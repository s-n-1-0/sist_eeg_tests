"""
trainデータの特定チャネルをスペクトログラムでプロットする。
(確認用)
"""
# %%
from  utils import spec
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
with open("tests/eeg_dataset_cnn/settings.json","r") as json_file:
    settings = json.load(json_file)
dataset_path = settings["dataset_path"]
csv = pd.read_csv(f"{dataset_path}/train/subj1_series2_data.csv")

# %%
channel = 1
data = csv.values
sig = np.array(data[:,channel],dtype=np.float32)
freqs,t,sp = spec.get_spectrogram(sig,500)
logsp = 10 * np.log10(sp)
plt.figure()
plt.pcolormesh(t,freqs,logsp, shading='auto')
plt.show()
# %%
