# %%
import pandas as pd
import os
import pyedflib
from typing import Callable, Optional
from numpy import ndarray
from labedf import csv2
from labedf.utilities import edf


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
        return df[["EDF","CSV","Type"]].values
    
def merge_all_csv2edf(file_settings:MergeAllCsv2EdfFileSettings,**csv2edf_options):
    if not os.path.exists(file_settings.build_dir_path):
        os.makedirs(file_settings.build_dir_path)
    edfcsv_filenames = file_settings.get_edfcsv_filenames()
    filenames:list[str] = []
    for i in range(edfcsv_filenames.shape[0]):
        edf_path = f"{file_settings.root_path}/edf/{edfcsv_filenames[i,0]}"
        csv_path = f"{file_settings.root_path}/csv/{edfcsv_filenames[i,1]}"
        filename = f"{file_settings.export_edf_name}_{i}_{edfcsv_filenames[i,2]}"
        csv2.merge_csv2edf(edf_path,csv_path,f"{file_settings.build_dir_path}/{filename}.edf",**csv2edf_options)
        filenames.append(filename)
    return filenames
def merge_all_unity2edf(file_settings:MergeAllCsv2EdfFileSettings,**csv2edf_options):
    if not os.path.exists(file_settings.build_dir_path):
        os.makedirs(file_settings.build_dir_path)
    edfcsv_filenames = file_settings.get_edfcsv_filenames()
    filenames:list[str] = []
    for i in range(edfcsv_filenames.shape[0]):
        edf_path = f"{file_settings.root_path}/edf/{edfcsv_filenames[i,0]}"
        csv_path = f"{file_settings.root_path}/csv/{edfcsv_filenames[i,1]}"
        filename = f"{file_settings.export_edf_name}_{i}"
        merge_unity2edf(edf_path,csv_path,f"{file_settings.build_dir_path}/{filename}.edf",**csv2edf_options)
        filenames.append(filename)
    return filenames
def merge_unity2edf(edf_path:str,
                csv_path:str,
                export_path:Optional[str] = None,
                marker_name:str = "Marker",
                sync_marker_name:str = "sync",
                end_marker_name:Optional[str]="__End__",
                end_marker_offset:float = 0,
                label_header_name:str = None,
                preprocessing_func:Optional[Callable[[list[ndarray]],list[ndarray]]] = None):
    """
    Unity csvとedfファイルをマージします。
    Args:
        edf_path (str): edf file path
        csv_path (str): csv file path
        export_path (str?): output file path. Defaults to None.(None is <edf_path + "-copy">)
        marker_name(str) : filter sender name(= "sender" value)
        sync_marker_name(str) : "response" value to synchronize files (None = 0 index)
        end_marker_name(str?) : annotation of marker_name end time
        end_marker_offset(float) : marker_name end time offset (seconds)
        label_header_name(str?) : label header name
        preprocessing_func(function) : preprocessing function
    """
    edf_dir_path = os.path.dirname(edf_path)
    edf_filename_path =  os.path.splitext(os.path.basename(edf_path))[0]
    if export_path is None:
        export_path = f"{edf_dir_path}/{edf_filename_path}_copy.edf"
    edf_reader = pyedflib.EdfReader(edf_path)
    rlab_dtype = {label_header_name:str} if not (label_header_name is None) else None
    rdf = pd.read_csv(csv_path,dtype=rlab_dtype)
    rdf_annos = rdf.values[:,0]
    rdf_sync_times = rdf.values[:,2]
    rdf_labels = rdf[label_header_name].values if label_header_name is not None else [None] * len(rdf_annos)

    edf_annos = edf.get_annotations(edf_reader)
    sync_edf_annos = [ea for ea in edf_annos if ea[0] == sync_marker_name]
    sync_rdf_annos_indexes = [i for i,r in enumerate(list(rdf_annos)) if r == sync_marker_name ]
    if len(sync_edf_annos) != len(sync_rdf_annos_indexes):
        raise Exception("Number of sync_marker_name in edf and csv files do not match")
    start_time_count:int = -1
    start_time_end:float = None
    results:list[tuple[float,float]] = []
    for ann, time_run,rdf_labels in zip(rdf_annos,rdf_sync_times,rdf_labels):
        if ann == sync_marker_name:
            start_time_count += 1
            start_time_end = rdf_sync_times[sync_rdf_annos_indexes[start_time_count]]
        offset_time_run = time_run - start_time_end + sync_edf_annos[start_time_count][1]
        if ann != marker_name:
            continue
        results.append((offset_time_run,rdf_labels))

    def copied_func(_ ,wedf:pyedflib.EdfWriter,signals:list[ndarray]):
        for otr,label in results:
            mn = marker_name
            if label is not None:
                mn += f"_{label}"
            wedf.writeAnnotation(otr,-1,mn)
            if not (end_marker_name is None):
                wedf.writeAnnotation(otr + end_marker_offset,-1,end_marker_name)
        if preprocessing_func is not None:
            return preprocessing_func(signals)
        return signals
    edf.copy(edf_reader,export_path,copied_func)