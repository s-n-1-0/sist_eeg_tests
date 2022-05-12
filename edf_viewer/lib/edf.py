import pyedflib
import math
import edf_viewer 
def get_all_signals(edf:pyedflib.EdfReader):
    """
    edfファイルから全チャンネルの波形を取得して返します。
    """
    ewavs = []
    for idx,_ in enumerate(edf.getSignalLabels()):
        ewavs.append(edf.readSignal(idx))
    return ewavs

def get_channel_length(edf:pyedflib.EdfReader):
    """
    チャンネル数を取得します。
    """
    return len(edf.getSignalHeaders())
def get_fs(edf:pyedflib.EdfReader):
    """
    edfファイルからサンプリングレートを取得します。
    """
    return edf.getSignalHeader(0)['sample_rate'] # = fs

def get_annotations(edf:pyedflib.EdfReader):
    """
    edfファイルからアノテーション情報を取得します。
    Returns:
       タプル(名前,経過秒,間隔,インデックス)を配列で返します。
    """
    annotations = edf.readAnnotations()
    fs = edf_viewer.get_fs(edf)
    rows:list[tuple[str,float,Any,int]] = [(name,time,duration,math.floor(time * fs))for time,duration,name in zip(annotations[0],annotations[1],annotations[2])]
    return rows
