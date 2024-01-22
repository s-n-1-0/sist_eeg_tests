# %%
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
from metrics import specificity
from joblib import load
csv_path = "./saves/mla_valid.csv"
valids = list(pd.read_csv(csv_path,header=None).to_numpy()[0,:])
sample_size = 750
batch_size = 32
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/merged_mla2.h5",valid_keys=[str(k) for k in valids])
# %% 1DCNN
model_path = "C:/_models/dec2/model-23.h5"
_,vgen = maker.make_generators(batch_size,pick_func=pfm.make_random_pick_func())
mode = "dl"
# %% EEGNET
model_path = "C:/_models/dec1/model-85.h5"
_,vgen = maker.make_2d_generators(batch_size,pick_func=pfm.make_random_pick_func())
mode ="dl"
# %% CSP LDA
model_path = "./cspldf.joblib"
_,vgen = maker.make_generators(None,pick_func=pfm.make_random_pick_func())
mode ="csplda"
# %%
if mode == "dl":
    loaded = keras.models.load_model(model_path,custom_objects={"specificity":specificity})
elif mode == "csplda":
    loaded = load(model_path) 
n = 100
acc_list = []
for i in range(n):
    y_pred = []
    y_true = []
    for data in  vgen():    
        x = data[0]
        if mode == "dl":
            Y_pred = loaded.predict(x)
            y_pred += [1 if y > 0.5 else 0 for y in Y_pred ]
        elif mode == "csplda":
            Y_pred = loaded.predict(x.transpose([0,2,1]))
            y_pred += list(Y_pred)
        y_true += list(data[1])
    cm = confusion_matrix(y_true, y_pred)
    acc = (cm[0,0]+cm[1,1])/np.sum(cm)
    print(i,cm)
    print(acc)
    acc_list.append(acc)
acc_list
# %%
acc_list = np.array(acc_list)
np.mean(acc_list),np.std(acc_list)
#1st
#1DCNN : (0.6312954876273654, 0.01345863209690109)
#EEGNet : (0.6458660844250363, 0.011716111904019416)
#CSP-LDA : (0.5563464337700146, 0.013535231747893315)
#2st
#1DCNN(OLD) : (0.6065882352941177, 0.010465444043667808)
#EEGNet(OLD) : (0.5943897058823531, 0.009554102340868672)
#CSPLDA(OLD) : (0.5986691176470589, 0.010211165249923512)
#1DCNN:
#   A:(0.6474264705882353, 0.00956306466998697)
#   B:(0.6533970588235296, 0.009712998728027162)
#   C:(0.6601617647058824, 0.009901318897851683)
#EEGNET:
#   A:(0.6455367647058823, 0.009070832733982551)
#   B:(0.6491544117647059, 0.008614167511745704)
#   C:(0.658264705882353, 0.010004389175165359)
# CSPLDA:
#   A : (0.587735294117647, 0.009335560061447817)
#   B : (0.5980441176470588, 0.008467843203089591)

# %%
