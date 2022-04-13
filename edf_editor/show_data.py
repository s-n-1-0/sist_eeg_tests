#%%
import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
EDF_PATH = "../edf_files/test2_0412_1.edf"
edf = pyedflib.EdfReader(EDF_PATH)

# %% edfプロパティを表示
headers = ["Prop", "Value"]
signalHeader = edf.getSignalHeader(0)
props = {
    "Ch Size":f"{len(edf.getSignalLabels())}ch",
    "Data Size":edf.getNSamples()[0],
    "Sample Rate":f"{signalHeader['sample_rate']}hz",
    "time [DataSize/SampleRate]":f"{edf.getFileDuration()}s"
}
print(tabulate(props.items(),headers,tablefmt="grid",colalign=('center','center')))
# %% edf波形データを表示
labels = edf.getSignalLabels()
signalHeader = edf.getSignalHeader(0)
times = np.arange(0,edf.getNSamples()[0]) / signalHeader['sample_rate']
plt.figure(figsize=(7,len(labels)*2))
for idx,label in enumerate(labels):
    plt.subplot(len(labels),1,idx + 1)
    plt.plot(times,edf.readSignal(idx),label=labels[idx])
plt.xlabel("Time(s)")
plt.show()
# %%
