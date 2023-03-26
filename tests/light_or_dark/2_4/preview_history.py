# %%
import pandas as pd
import matplotlib.pyplot as plt
from utils.history import plot_history
# %%
df = pd.read_csv("./saves/lord/2_4/count/history.csv")
plot_history(df,metrics=["binary_accuracy"],
             is_loss=False,
             title="カウント/非カウント 二値分類",
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
