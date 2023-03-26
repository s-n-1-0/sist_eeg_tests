# %%
from labedf import edf2,set2
import numpy as np
from pyedflib import EdfReader
from utils import edf as myedf,edflab as myedflab,signals_standardization,lowpass_filter
DATASET_DIR_PATH = "./dataset/lord2/train"
file_settings = myedflab.MergeAllCsv2EdfFileSettings(DATASET_DIR_PATH + "/ペア.csv",list_encoding="ansi")
edfcsv_filenames = file_settings.get_edfcsv_filenames()
with EdfReader(f"{DATASET_DIR_PATH}/edf/{edfcsv_filenames[0,0]}") as er:
    fs = int(myedf.get_fs(er))

# %% merge csv,edf
filenames = myedflab.merge_all_csv2edf(file_settings,label_header_name="LorD",marker_names=["Marker","Wait"],marker_offset=None)
filenames

# %% set to hdf
export_path = f"{DATASET_DIR_PATH}/ex_lord.h5"
def after_preprocessing(signals:np.ndarray,label:str):
    if label != "dark" and label != "light":
        return signals
    return signals_standardization(signals)
for i ,filename in enumerate(filenames):
    set2.merge_set2hdf(f"{file_settings.root_path}/pre/{filename}.set",
    export_path,
    labels=["dark","light"],
    is_groupby=True,
    is_overwrite= i != 0,
    )


# %%
