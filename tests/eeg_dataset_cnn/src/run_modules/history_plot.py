# %%
from typing import Union
import matplotlib.pyplot as plt
import json
import pandas as pd
def history_plot(input:Union[str,dict]):
    if isinstance(input,str):
        history = pd.read_csv(input)
    else:
        history = input
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
    plt.plot(epochs, binary_acc, linestyle = "solid", label= 'train binary-acc')
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
if __name__ == '__main__':
    with open("tests/eeg_dataset_cnn/src/settings.json","r") as json_file:
        settings = json.load(json_file)
        work_path =  settings["work_path"]

    csv_path = f"{work_path}/dest/model_2022_06_01_00_07_history.csv"
    history_plot(csv_path)
# %%
