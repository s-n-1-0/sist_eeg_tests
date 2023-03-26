# %%
from typing import Callable
import numpy as np
from pyedflib import EdfReader
import mne
import h5py
from utils import edf as myedf,edflab as myedflab,signals_standardization
DATASET_DIR_PATH = "./dataset/lord2/train"
file_settings = myedflab.MergeAllCsv2EdfFileSettings(DATASET_DIR_PATH + "/ペア.csv",list_encoding="ansi")
edfcsv_filenames = file_settings.get_edfcsv_filenames()
with EdfReader(f"{DATASET_DIR_PATH}/edf/{edfcsv_filenames[0,0]}") as er:
    fs = int(myedf.get_fs(er))

# %% merge csv,edf
filenames = myedflab.merge_all_csv2edf(file_settings,label_header_name="LorD",marker_names=["Marker","Wait"],marker_offset=None)
filenames

# %% set to hdf
dark_count = 0
light_count = 0
c = 0
nc = 0
def merge_set2hdf(edf_path:str,
    export_path:str,
    key_type:str,
    labels:list[str],
    marker_names:list[str] = ["Marker"],
    is_overwrite:bool = False,
    is_groupby:bool = False,
    preprocessing_func:Callable[[np.ndarray,str],np.ndarray] = None,
    ):
    """Split the edf file by annotation and save it in the hdf file.
     Args:
        edf_path : read edf path
        export_path : write hdf path
        is_overwrite : overwrite the edf file
        is_groupby : grouping
        preprocessing_func(function[[signals,label],ndarray]?) : Processes each segmented data. ndarray : ch × annotation range
    """

    epochs = mne.io.read_epochs_eeglab(edf_path)
    with h5py.File(export_path, mode='r+' if is_overwrite else 'w') as f:
        def write_hdf(marker_name:str,label:str,new_label:str):
            global dark_count, light_count,c,nc
            data = epochs.get_data(item=f"{marker_name}__{label}")
            if label == "dark":
                dark_count += data.shape[0]
            elif label == "light":
                light_count += data.shape[0]
            if new_label == "c":
                c += data.shape[0]
            elif new_label == "nc":
                nc += data.shape[0]
            ann_group = f.require_group("/annotations")
            for idx in range(data.shape[0]):
                data_ch = data[idx,:,:]
                if not(preprocessing_func is None):
                    data_ch = preprocessing_func(data_ch,label)
                if is_groupby:
                        local_group = ann_group.require_group("./" + marker_name)
                        counter = local_group.attrs.get("count")
                        counter = 0 if counter is None else counter + 1
                        local_group.attrs["count"] = counter
                        d = local_group.create_dataset(f"{counter}",data_ch.shape,data=data_ch)
                        d.attrs["label"] = new_label
                else:
                        d = ann_group.create_dataset(f"{idx}.{marker_name}",data_ch.shape,data=data_ch)
                        d.attrs["label"] = new_label
        for marker_name in marker_names:
            for label in labels:
                new_label = "c" if (label == "dark" and key_type == "dc") or \
                (label == "light" and key_type == "lc") else "nc"
                write_hdf(marker_name,label,new_label)
def preprocessing(signals:np.ndarray,label:str):
    if label != "dark" and label != "light":
        return signals
    return signals_standardization(signals)
export_path = f"{DATASET_DIR_PATH}/ex_count.h5"
for i ,filename in enumerate(filenames):
    dataset_type = filename.split("_")[-1]
    merge_set2hdf(f"{file_settings.root_path}/pre/{filename}.set",
    export_path,
    dataset_type,
    labels=["dark","light"],
    is_groupby=True,
    is_overwrite= i != 0,
    #preprocessing_func=preprocessing
    )
print(f"dark : {dark_count}")
print(f"light : {light_count}")
print(f"counts : {c}")
print(f"not counts : {nc}")

# %%
