# %%
import os
import pyedflib
import edf_viewer as myedf
import pandas as pd
import numpy as np
EDF_PATH = "edf_files/test4/kurihara_0510_try2.edf"
CSV_PATH = "tests/test4/test4_try2--2022-05-10--17_04_54.csv"
#----------
edf_dir_path = os.path.dirname(EDF_PATH)
edf_filename_path =  os.path.splitext(os.path.basename(EDF_PATH))[0]
out_edf_path = f"{edf_dir_path}/{edf_filename_path}_copy.edf" 
edf_reader = pyedflib.EdfReader(EDF_PATH)
ch = myedf.get_channel_length(edf_reader)

# %% 【前のセル実行必須】csvからアノテーションを生成
annos = myedf.get_annotations(edf_reader)
csv = pd.read_csv(CSV_PATH,usecols=[0,3,7,8,11,13]).values
start_time_commits = csv[0,4]
marker_indexes = np.where(csv[:,0] == "Marker")[0]
time_runs = csv[marker_indexes,5]
offset_times = ((time_runs - start_time_commits) / 1000.0) + annos[1][1]
# %% 【前のセル実行必須】
def copied_func(redf:pyedflib.EdfReader,wedf:pyedflib.EdfWriter):
    for ot in offset_times:
        wedf.writeAnnotation(ot,-1,"action")
myedf.copy(edf_reader,out_edf_path,copied_func)
# %%
