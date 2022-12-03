
# %%
import pandas as pd
import os
import numpy as np
import json
from scipy.signal import lfilter, butter
IS_FILTER = True
FN = 4
LOWCUT = 0.2
HIGHCUT = 50
with open("tests/eeg_dataset_cnn/settings.json","r") as json_file:
    settings = json.load(json_file)
train_path = os.path.join(settings["dataset_path"],"train")
work_path = settings["work_path"]
fs = settings["fs"]
# %%
def read_csv(x_path:str, y_path:str):
    def make_butter_bandpass():
        nyq = 0.5 * fs
        cutoff = [LOWCUT / nyq, HIGHCUT / nyq]
        b, a = butter(FN, cutoff, btype="bandpass")
        return b, a
    b,a = make_butter_bandpass()
    xcsv = pd.read_csv(x_path)
    ycsv = pd.read_csv(y_path)
    x = xcsv.iloc[:,1:].values.astype(np.float32)
    y = ycsv.iloc[:,1:].values.astype(np.float32)
    x = lfilter(b, a, x, axis=0)
    return x, y
def read_csv_in_folder(folder_path:str,filename:str):
    data_path = os.path.join(folder_path, filename)
    filekey = filename.split(".")[0].split("_")[:-1]
    events_path = os.path.join(folder_path, f"{'_'.join(filekey)}_events.csv")
    return read_csv(data_path, events_path)

trainset = [read_csv_in_folder(train_path,filename) for filename in os.listdir(train_path) if 'data' in filename]
# %% split
def concatenate(lst:list,interpolate:bool):
    """結合部補間付き"""
    def destructive_interpolate(condata:np.ndarray, p:int, size=32):
        mean = 0.5 * (condata[p-1] + condata[p])
        region = condata[p-size:p+size] - mean
        region[:size] *= np.linspace(1,0,size)[:,None]
        region[size:] *= np.linspace(0,1,size)[:,None]
        condata[p-size:p+size] = region + mean
    
    result = np.concatenate(lst,axis=0).astype(np.float32)
    if interpolate:
        p = 0
        for s in lst[:-1]:
            p += len(s)
            destructive_interpolate(result, p)
    return result
def std(data:np.ndarray)->np.ndarray:
    m = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - m ) /std
#x,yそれぞれ取得
train_x = [t[0] for t in trainset]
train_y = [t[1] for t in trainset]
#検証用を分離
valid_x = concatenate(train_x[-3:],interpolate=True)
valid_y = concatenate(train_y[-3:],interpolate=False)
test_x = concatenate(train_x[:2],interpolate=True)
test_y = concatenate(train_y[:2],interpolate=False)
train_x = concatenate(train_x[2:-3],interpolate=True)
train_y  = concatenate(train_y[2:-3],interpolate=False)

train_x = std(train_x)
valid_x = std(valid_x)
test_x = std(test_x)

# %% save
np.savez_compressed(f"{work_path}/dest/prep.npz",train_x = train_x,train_y = train_y,valid_x=valid_x,valid_y=valid_y,test_x=test_x,test_y=test_y)
# %%
