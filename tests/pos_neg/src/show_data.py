# %% import
import glob
import re
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from itertools import groupby
import pyedflib
import edf_viewer
from edf_viewer import spec
import pandas as pd
from datetime import datetime
import math
from scipy import signal
# %% config
EDF_PATH = "edf_files/pos_neg/posneg_0421_1.edf"
CSV_PATH = "tests/pos_neg/pos_neg--2022-04-21--16_14_07.csv"
IMG_DIR_PATH = "tests/pos_neg/images"
POS_GROUP_NAMES = ["B","P","Q"]
NEG_GROUP_NAMES = ["A","H","Z"]
NEU_GROUP_NAMES = ["N"]
# %% edfファイルの取得
edf = pyedflib.EdfReader(EDF_PATH)
fs = edf_viewer.get_fs(edf)#サンプリングレート
annos = edf_viewer.get_annotations(edf)
# %% 画像データの表示
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
# %% edfからeegを取得し実験部分以外を切り捨て
all_signals = edf_viewer.get_all_signals(edf)
#use_colsによって実験開始までの行データをスキップ
working_range_csv = pd.read_csv(CSV_PATH,skiprows=4,usecols=[0,3,7,8,11,12]).values
#シークエンスタイトルを削除
working_range_csv = working_range_csv[np.where(working_range_csv[:,0] == 'goodbad')]
goodbad_anss = working_range_csv[:,3]
run_times = working_range_csv[:,4] / 1000 #秒に変換
end_times = working_range_csv[:,5] / 1000 #秒に変換
#開始アノテーションを0秒とした値
start_anno = annos[1] # = 'sync'アノテーション
offset_run_times = np.asarray([rt - run_times[0] for rt in run_times])
offset_run_time_indexes = [math.floor(ort * fs) for ort in offset_run_times]
fixed_offset_run_times = np.asarray(offset_run_times) + start_anno[1]
fixed_offset_run_time_indexes = np.asarray(offset_run_time_indexes) + start_anno[3]

# %% 実験結果(goodbad回答)をプロット
freqs,t,_ = signal.stft(all_signals[0],fs=fs, detrend=False, window='hanning', noverlap=128)
especs = spec.get_spectrograms(all_signals,fs)
log_especs = 10 * np.log10(especs)
plt.figure()
plt.title("avg ch (ans line)")
plt.pcolormesh(t,freqs,np.mean(log_especs,axis=0), shading='auto')
_,r,*_ = start_anno
plt.vlines(r, 0, freqs[-1],colors='#FF4C4C',linestyle='dashed', linewidth=3)
def ans2color(ans:str):
    if ans == "good":
        return "#FF4C4C"
    elif ans == "bad":
        return "#4c8bff"
    else:
        return "white"
for ans,fot in zip(goodbad_anss,fixed_offset_run_times):
    plt.vlines(fot, 0, freqs[-1],colors=ans2color(ans),linestyle='dashed', linewidth=3)
plt.show()
plt.show()

# %%
