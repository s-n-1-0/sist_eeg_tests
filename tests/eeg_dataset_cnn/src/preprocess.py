
# %%
import pandas as pd
import os
import numpy as np
import json
with open("tests/eeg_dataset_cnn/src/settings.json","r") as json_file:
    settings = json.load(json_file)
train_path = os.path.join(settings["dataset_path"],"train")
work_path = settings["work_path"]
# %%
def read_csv(x_path:str, y_path:str):
    xcsv = pd.read_csv(x_path)
    ycsv = pd.read_csv(y_path)
    x = xcsv.iloc[:,1:].values
    y = ycsv.iloc[:,1:].values
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
def norm(data:np.ndarray)->np.ndarray:
    m = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - m ) /std
#x,yそれぞれ取得
train_x = [t[0] for t in trainset]
train_y = [t[1] for t in trainset]
#検証用を分離
valid_x = concatenate(train_x[-5:],interpolate=True)
valid_y = concatenate(train_y[-5:],interpolate=False)
train_x = concatenate(train_x[:-5],interpolate=True)
train_y  = concatenate(train_y[:-5],interpolate=False)
train_x = norm(train_x)
valid_x = norm(valid_x)

# %% save
np.savez_compressed(f"{work_path}/tmp/prep.npz",train_x = train_x,train_y = train_y,valid_x=valid_x,valid_y=valid_y)
# %%
