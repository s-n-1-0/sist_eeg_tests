# %% import
import glob
import re
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from itertools import groupby
# %% config
IMG_DIR_PATH = "tests/pos_neg/images"
POS_GROUP_NAMES = ["B","P","Q"]
NEG_GROUP_NAMES = ["A","H","Z"]
NEU_GROUP_NAMES = ["N"]
# %% 使用データの表示
files = glob.glob(f"{IMG_DIR_PATH}/*.bmp")
def get_file_data(f:str):
    ex = re.search(r"[^\\]+$",f).group()
    ex_group = ex[0]
    img = np.asarray(Image.open(f))
    return (f,ex,ex_group,img)
def plot_groups(title:str,all_groups:list,group_names:list):
    plt.figure()
    groups = []
    _ = [groups.append(g) for g in all_groups if group_names.count(g[0]) > 0]
    groups_size = len(groups)
    max_images = max([len(g[1]) for g in groups])
    fig, ax = plt.subplots(groups_size,max_images, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(title,fontsize=32)
    for i, exg in enumerate(groups):
        images = [f[3] for f in exg[1]]
        fig.subplots_adjust(hspace=0, wspace=0)
        for j in range(len(images)):
            asp =  ax[i,j] if groups_size > 1 else ax[j]
            asp.xaxis.set_major_locator(plt.NullLocator())
            asp.yaxis.set_major_locator(plt.NullLocator())
            asp.imshow(images[j], cmap="bone")
    plt.show()
exs = [get_file_data(f) for f in files]
exs.sort(key=lambda v: v[2])
exs_groups = [(key,list(group)) for key, group in groupby(exs, key=lambda m: m[2])]
plot_groups("Positive Images",exs_groups,POS_GROUP_NAMES)
plot_groups("Negative Images",exs_groups,NEG_GROUP_NAMES)
plot_groups("Neutral Images",exs_groups,NEU_GROUP_NAMES)
# %%
