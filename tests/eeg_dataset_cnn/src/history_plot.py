# %%
import matplotlib.pyplot as plt
import json
import pandas as pd
with open("tests/eeg_dataset_cnn/src/settings.json","r") as json_file:
    settings = json.load(json_file)
    work_path =  settings["work_path"]

SAVE_CSV_PATH = f"{work_path}/dest/model_2022_05_31_23_01_history.csv"

# %%
history = pd.read_csv(SAVE_CSV_PATH)
total_acc = history['total_acc']
val_total_acc = history['val_total_acc']
binary_acc = history['binary_acc']
val_binary_acc = history['val_binary_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(total_acc))

plt.figure(figsize=(12, 10))
plt.rcParams['font.size'] = 25

plt.plot(epochs, total_acc, linestyle = "solid", label = 'train total-acc')
plt.plot(epochs, val_total_acc, linestyle = "solid", label= 'valid total-acc')
plt.plot(epochs, binary_acc, linestyle = "solid", label= 'valid binary-acc')
plt.plot(epochs, val_binary_acc, color = "green", linestyle = "solid", label= 'valid binary-acc')
plt.plot(epochs, loss,linestyle = "solid" ,label = 'train loss')
plt.plot(epochs, val_loss,linestyle = "solid" , label= 'valid loss')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#plt.ylim(0,1)
plt.grid()
plt.show()
#plt.savefig("save_path")
plt.close()
# %%
