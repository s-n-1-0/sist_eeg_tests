# %%
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
from metrics import specificity
csv_path = "./saves/mla_valid.csv"
valids = list(pd.read_csv(csv_path,header=None).to_numpy()[0,:])
sample_size = 750
batch_size = 32
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/merged_mla2.h5",valid_keys=[str(k) for k in valids])
# %% 1DCNN
model_path = "C:/_models/dec2/model-23.h5"
_,vgen = maker.make_generators(batch_size,pick_func=pfm.make_random_pick_func())
# %% EEGNET
model_path = "C:/_models/dec1/model-85.h5"
_,vgen = maker.make_2d_generators(batch_size,pick_func=pfm.make_random_pick_func())
# %%
loaded = keras.models.load_model(model_path,custom_objects={"specificity":specificity})
loaded
n = 100
acc_list = []
for i in range(n):
    y_pred = []
    y_true = []
    for data in  vgen():
        Y_pred = loaded.predict(data[0])
        y_pred += [1 if y > 0.5 else 0 for y in Y_pred ]
        y_true += list(data[1])
    cm = confusion_matrix(y_true, y_pred)
    acc = (cm[0,0]+cm[1,1])/np.sum(cm)
    print(cm)
    print(acc)
    acc_list.append(acc)
acc_list
# %%
acc_list = np.array(acc_list)
np.mean(acc_list),np.std(acc_list)
#1DCNN : (0.6056250000000001, 0.010647381222586754)
#EEGNet : (0.5920758928571428, 0.00807103061137033)
# %%
