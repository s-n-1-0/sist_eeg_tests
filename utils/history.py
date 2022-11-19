from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
def plot_history(h:Any,metrics:list[str]):
    """
    h = history.history
    """
    loss = h['loss']
    val_loss = h['val_loss']
    #train_acc = h['binary_accuracy']
    #val_total_acc = h['val_binary_accuracy']
    epochs = range(len(loss))

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 25
    plt.plot(epochs, loss,linestyle = "solid" ,label = 'loss')
    plt.plot(epochs, val_loss,linestyle = "solid" , label= 'valid loss')
    for mn in metrics:
        acc = h[mn]
        val_acc = h['val_'+mn]
        plt.plot(epochs, acc, linestyle = "solid", label = mn)
        plt.plot(epochs, val_acc, linestyle = "solid", label= 'valid '+mn)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
    plt.grid()
    plt.show()
    plt.close()

def save_history(dir_path:str,h:Any):
    """
    h = history.history
    """
    hdf = pd.DataFrame(h)
    if dir_path[-1] != "/":
        dir_path = dir_path + "/"
    hdf.to_csv(f"{dir_path}history.csv")