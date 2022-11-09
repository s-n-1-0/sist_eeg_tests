# %%
import os
from labedf import csv2,edf2
PROJECT_DATA_DIR_PATH = "./edf_files/lord"
# %% merge csv,edf
build_dir_path = f"{PROJECT_DATA_DIR_PATH}/build"
if not os.path.exists(build_dir_path):
    os.makedirs(build_dir_path)
file_names = [fp.split(".")[0] for fp in os.listdir(f"{PROJECT_DATA_DIR_PATH}/edf")]
for file_name in file_names:
    edf_path = f"{PROJECT_DATA_DIR_PATH}/edf/{file_name}.edf"
    csv_path = f"{PROJECT_DATA_DIR_PATH}/csv/{file_name}.csv"
    csv2.merge_csv2edf(edf_path,csv_path,f"{build_dir_path}/{file_name}.edf")
# %% to hdf
export_path = f"{PROJECT_DATA_DIR_PATH}/ex.h5"
for i ,file_name in enumerate(file_names):
    edf2.split_annotations_edf2hdf(f"{build_dir_path}/{file_name}.edf",export_path,is_groupby=True,is_overwrite= i != 0)

# %%
