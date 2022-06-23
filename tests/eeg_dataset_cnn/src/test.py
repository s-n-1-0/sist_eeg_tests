# %%
import json
from keras import models
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"tests/eeg_dataset_cnn/src"))
from run_modules import EEGDataset,Maxout,total_acc,binary_acc,recall,precision
# %%
"""
データセットにフィルターを適用しない場合: prep_non_filter.npz
フィルター適用訓練済みモデルを使用しない場合: model_2022_06_01_00_07
"""

with open("tests/eeg_dataset_cnn/settings.json","r") as json_file:
    settings = json.load(json_file)
work_path = settings["work_path"]

saved_path = f"{work_path}/dest/model_2022_06_01_00_07"
dataset = EEGDataset()
dataset.read_dataset(filename="prep.npz")

# %%
batch_size = 4
test_gen = dataset.make_valid_generator(4096,batch_size)
model = models.load_model(saved_path,custom_objects={"Maxout":Maxout,"total_acc":total_acc,"binary_acc":binary_acc,"recall":recall,"precision":precision})
results = model.evaluate(test_gen,batch_size=batch_size)
print(results)
# %%
