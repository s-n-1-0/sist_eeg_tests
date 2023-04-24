# %%
import pandas as pd
import matplotlib.pyplot as plt
def plot_history(h,metrics:list[str],
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

    plt.figure(figsize=(20, 10))
    plt.rcParams['font.size'] = 25
    #plt.ylim([0.4,0.7])
    if is_loss:
        plt.plot(epochs, loss,linestyle = "solid" ,label = 'loss')
        plt.plot(epochs, val_loss,linestyle = "dash" , label= 'valid loss')
    else:
        print("")
        #plt.ylabel("精度" ,fontname="MS Gothic",fontsize=20)
    for mn in metrics:
        acc = h[mn]
        val_acc = h['val_'+mn]
        plt.plot(epochs, acc, linestyle = "solid", label = "学習精度",color="b",lw=5)
        plt.plot(epochs, val_acc, linestyle = "dashdot", label= 'テスト精度',color="r",lw=5)
    if title is not None:
        plt.title(title,fontname="MS Gothic")
    #plt.xlabel("エポック数(学習回数)" ,fontname="MS Gothic",fontsize=20)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=1,prop={"family":"MS Gothic","size": 50})
    plt.ylim([0.4,0.7])
    plt.grid()
    plt.show()
    plt.close()
# %%
df = pd.read_csv("./saves/lord/2_4/count/history.csv")
plot_history(df,metrics=["binary_accuracy"],
             is_loss=False,
             title="",
             bbox_to_anchor=(1,0),
             loc="lower right")
# %%
df = pd.read_csv("./saves/lord/2_4/lord/history.csv")
plot_history(df,metrics=["binary_accuracy"],
             is_loss=False,
             title="明るい/暗い 二値分類",
             bbox_to_anchor=(1,0),
             loc="lower right")
# %%
