# %%
import pandas as pd
import matplotlib.pyplot as plt
def plot_history(h,metrics:list[str],
                 bbox_to_anchor=(1, 1),
                 loc='upper right'):
                 
    """
    h = history.history
    """
    loss = h['loss']
    epochs = range(len(loss))

    plt.figure(figsize=(20, 10))
    plt.rcParams['font.size'] = 25
    for mn in metrics:
        acc = h[mn]
        val_acc = h['val_'+mn]
        plt.plot(epochs, acc, linestyle = "solid", label = "学習精度",color="b",lw=5)
        plt.plot(epochs, val_acc, linestyle = "dashdot", label= 'テスト精度',color="r",lw=5)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=1,prop={"family":"MS Gothic","size": 50})
    plt.ylim([0.4,0.7])
    plt.grid()
    plt.show()
    plt.close()
# %%
df = pd.read_csv("./saves/lord/2_4/count/history.csv")
plot_history(df,metrics=["binary_accuracy"],
             bbox_to_anchor=(1,0),
             loc="lower right")
# %%
df = pd.read_csv("./saves/lord/2_4/lord/history.csv")
plot_history(df,metrics=["binary_accuracy"],
             bbox_to_anchor=(1,0.5),
             loc="lower right")
# %%
