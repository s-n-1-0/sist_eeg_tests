# %%
import pandas as pd
from labedf import csv2
import os

class MergeAllCsv2EdfFileSettings():
    def __init__(self,list_path:str,list_encoding=None):
        root_path = os.path.dirname(list_path)
        self.root_path = root_path
        self.list_path =  list_path
        self.list_encoding = list_encoding
        self.build_dir_path = f"{root_path}/build"
        self.export_edf_name = "merged"
    # %% CSVとEDFファイルのペア情報を取得する
    def get_edfcsv_filenames(self):
        df = pd.read_csv(self.list_path,encoding=self.list_encoding)
        return df[["EDF","CSV"]].values
    
def merge_all_csv2edf(file_settings:MergeAllCsv2EdfFileSettings,**csv2edf_options):
    if not os.path.exists(file_settings.build_dir_path):
        os.makedirs(file_settings.build_dir_path)
    edfcsv_filenames = file_settings.get_edfcsv_filenames()
    filenames:list[str] = []
    for i in range(edfcsv_filenames.shape[0]):
        edf_path = f"{file_settings.root_path}/edf/{edfcsv_filenames[i,0]}"
        csv_path = f"{file_settings.root_path}/csv/{edfcsv_filenames[i,1]}"
        filename = f"{file_settings.export_edf_name}_{i}"
        csv2.merge_csv2edf(edf_path,csv_path,f"{file_settings.build_dir_path}/{filename}.edf",**csv2edf_options)
        filenames.append(filename)
    return filenames
