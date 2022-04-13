#%%
import pyedflib
import matplotlib.pyplot as plt
EDF_PATH = "../edf_files/test2_0412_1.edf"
edf = pyedflib.EdfReader(EDF_PATH)

# %% edf波形データを表示
labels = edf.getSignalLabels()
plt.figure(figsize=(7,len(labels)*2))
for idx,label in enumerate(labels):
    plt.subplot(len(labels),1,idx + 1)
    plt.plot(edf.readSignal(idx)[0:500],label=labels[idx])
plt.show()
# %%
