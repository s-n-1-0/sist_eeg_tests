# %% import
import glob
import re
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from itertools import groupby
import pyedflib
import utils
import pandas as pd
from datetime import datetime
import math
from scipy import signal
from tabulate import tabulate
# %% 【必須1】config
EDF_PATH = "edf_files/pos_neg/posneg_0421_1.edf"
CSV_PATH = "tests/pos_neg/pos_neg--2022-04-21--16_14_07.csv"
IMG_DIR_PATH = "tests/pos_neg/images"
RESPONDENT_PRIORITY = True
POS_GROUP_NAMES = ["B","P","Q"]
NEG_GROUP_NAMES = ["A","H","Z"]
NEU_GROUP_NAMES = ["N"]
# %% 【必須2】edfファイルの取得
edf = pyedflib.EdfReader(EDF_PATH)
fs = utils.get_fs(edf)#サンプリングレート
annos = utils.get_annotations(edf)
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
all_signals = utils.get_all_signals(edf)
#use_colsによって実験開始までの行データをスキップ
"""
0:title -> 0
3:timestamp -> 1
6:file_name -> 2
7:response -> 3
8:time_commit -> 4
11:time_end -> 5
13:time_run -> 6
"""
working_range_csv = pd.read_csv(CSV_PATH,skiprows=4,usecols=[0,3,7,8,11,13]).values
#goodbad回答以外を削除
working_range_csv = working_range_csv[np.where(working_range_csv[:,0] == 'goodbad')]
file_names = working_range_csv[:,2]
goodbad_anss = working_range_csv[:,3]
if not RESPONDENT_PRIORITY:
    def group2goodbad(group:str):
        if POS_GROUP_NAMES.count(group) > 0:
            return "good"
        elif NEG_GROUP_NAMES.count(group) > 0:
            return "bad"
        else:
            return "none"
    goodbad_anss = [group2goodbad(fname[0]) for fname in file_names]
run_times = working_range_csv[:,5] / 1000 #秒に変換
end_times = working_range_csv[:,4] / 1000 #秒に変換
#開始アノテーションを0秒とした値
start_anno = annos[1] # = 'sync'アノテーション
#run_time[0]を開始点とするためlab.js内オフセット
offset_run_times = np.asarray([rt - run_times[0] for rt in run_times])
offset_end_times = np.asarray([et - run_times[0] for et in end_times])
offset_run_time_indexes = [math.floor(ort * fs) for ort in offset_run_times]
offset_end_time_indexes = [math.floor(ort * fs) for ort in offset_end_times]
#eeg基準に修正
fixed_offset_run_times = np.asarray(offset_run_times) + start_anno[1]
fixed_offset_run_time_indexes = np.asarray(offset_run_time_indexes) + start_anno[3]
fixed_offset_end_times = np.asarray(offset_end_times) + start_anno[1]
fixed_offset_end_time_indexes = np.asarray(offset_end_time_indexes) + start_anno[3]

#各回答ごとにeegをスプリット
freqs,t,all_specs = utils.spec.get_spectrograms(all_signals,fs)
ans_trange_indexes = [np.where((t>=rt) & (t<=et))[0] for rt,et in zip(fixed_offset_run_times,fixed_offset_end_times)]
ans_not_trange_indexes = [np.where((t<rt) | (t>et))[0] for rt,et in zip(fixed_offset_run_times,fixed_offset_end_times)]
def zero_padding_spec(sp:np.ndarray,zero_range:np.ndarray):
    """
    スペクトログラムの指定した範囲を0埋めします。
    """
    new_sp = np.copy(sp)
    new_sp[:,zero_range] = 0
    return new_sp
ans_specs = [all_specs[...,at] for at in ans_trange_indexes]
ans_padding_specs = [[zero_padding_spec(sp,at) for sp in all_specs] for at in ans_not_trange_indexes]
#ans_ch_especs 
# %% 【前のセル実行必須】実験結果(goodbad回答)をプロット

log_especs = 10 * np.log10(all_specs)
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

# %% 各回答ごとに波形情報抽出
alpha_freq_indexes,alpha_freqs = utils.pass_range.pass_range(freqs,8,13)
beta_freq_indexes,beta_freqs = utils.pass_range.pass_range(freqs,13,24)
ans_alpha_specs = []
ans_beta_specs = []
# a,b,a/bのタプル
total_results = []
good_results = []
bad_results = []
none_results = []
for ans,asps in zip(goodbad_anss,ans_specs): # aspsize= ch x freqs x t
    asp = asps[:,alpha_freq_indexes,:]
    bsp = asps[:,beta_freq_indexes,:]
    ans_alpha_specs.append(asp)
    ans_beta_specs.append(bsp)
    a = np.mean(asp)
    b = np.mean(bsp)
    raito = b / a
    total_results.append((a,b,raito))
    if ans == "good":
        good_results.append((a,b,raito))
    elif ans == "bad":
        bad_results.append((a,b,raito))
    elif ans == "none":
        none_results.append((a,b,raito))
    #print(f"{ans}:{np.mean(bsp)/np.mean(asp)}")
def get_mean_resuls(results:list):
    return (np.mean([r[0] for r in results]),np.mean([r[1] for r in results]),np.mean([r[2] for r in results]))
headers = ["Group", "α","β","β/α"]
mean_good_results = get_mean_resuls(good_results)
mean_bad_results = get_mean_resuls(bad_results)
mean_none_results = get_mean_resuls(none_results)

print("チャンネル平均")
print(tabulate([("Good",) + mean_good_results,("Bad",) + mean_bad_results,("None",) + mean_none_results],headers,tablefmt="grid",colalign=('center','center')))
""" 各回答のα帯のスペクトログラム(表示できるけど拡大しすぎてよくわからなくなってる)
for ati,aasp in zip(ans_trange_indexes,ans_alpha_specs):
    at = t[ati]
    log_especs = 10 * np.log10(aasp)
    plt.figure()
    plt.title("avg ch (ans line)")
    print(log_especs.shape)
    plt.pcolormesh(at,alpha_freqs,np.mean(log_especs,axis=0), shading='flat')
    plt.show()
    plt.show()
"""
plt.title("avg ch : α,β,β/α")
plt.plot(range(len(goodbad_anss)),[tr[0] for tr in total_results],label="α")
plt.plot(range(len(goodbad_anss)),[tr[1] for tr in total_results],label="β")
plt.plot(range(len(goodbad_anss)),[tr[2] for tr in total_results],label="β/α")
plt.xlabel("count")
plt.legend()
def ans2color(ans:str):
    if ans == "good":
        return "#FF4C4C"
    elif ans == "bad":
        return "#4c8bff"
    else:
        return "black"
for idx,ans in enumerate(goodbad_anss):
    plt.vlines(idx, 0, 5,colors=ans2color(ans),alpha = 0.2,linestyle='dashed', linewidth=3)
plt.show()
# %%
