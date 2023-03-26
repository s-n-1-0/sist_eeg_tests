from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
def plot_history(h:Any,metrics:list[str],
                 is_loss:bool = True,
                 title:str=None,
                 bbox_to_anchor=(1, 1),
                 loc='upper right'):
                 
    """
    h = history.history
    """
    loss = h['loss']
    val_loss = h['val_loss']
    epochs = range(len(loss))

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 25
    #plt.ylim([0.4,0.7])
    if is_loss:
        plt.plot(epochs, loss,linestyle = "solid" ,label = 'loss')
        plt.plot(epochs, val_loss,linestyle = "dashdot" , label= 'valid loss')
    else:
        plt.ylabel("精度" ,fontname="MS Gothic")
    for mn in metrics:
        acc = h[mn]
        val_acc = h['val_'+mn]
        plt.plot(epochs, acc, linestyle = "solid", label = mn)
        plt.plot(epochs, val_acc, linestyle = "dashdot", label= 'valid '+mn)
    if title is not None:
        plt.title(title,fontname="MS Gothic")
    plt.xlabel("エポック数(学習回数)" ,fontname="MS Gothic")
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=1)
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