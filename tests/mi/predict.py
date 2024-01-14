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
#1DCNN(OLD) : (0.6056250000000001, 0.010647381222586754)
#EEGNet(OLD) : (0.5920758928571428, 0.00807103061137033)
#1DCNN:
#   A:(0.6494940476190476, 0.010154371991900175)
#   B:(0.6532663690476191, 0.010192363406378396)
#   C:(0.6607142857142857, 0.009044387737513543)
#EEGNET:
#   A:(0.6448586309523808, 0.010756414672879515)
#   B:(0.6469568452380954, 0.01009357295394528)
#   C:(0.6586681547619048, 0.009123234475514141)
# CSPLDA:
#   A : 0.5876617647058824, 0.009372955371041903
#   B : (0.5989926470588236, 0.008427160299255169)
#  OLD : (0.5993749999999999, 0.00961936427824869)
# %%
