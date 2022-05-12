from pyclbr import Function
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

def copy(redf:pyedflib.EdfReader,copy_path:str,copied_func:Function = None):
    """
    redfの内容をコピーパスへコピーします。
    copied_funcの第一引数にredf,第二引数にwedfを返します。
    """
    ch = get_channel_length(redf)
    with pyedflib.EdfWriter(copy_path,ch) as wedf:
        header = redf.getHeader()
        header["birthdate"] = ""
        annos = redf.readAnnotations()
        wedf.setHeader(header)
        wedf.setSignalHeaders(redf.getSignalHeaders())
        wedf.writeSamples(get_all_signals(redf))
        for i,_ in enumerate(annos[0]):
            wedf.writeAnnotation(annos[0][i],annos[1][i],annos[2][i])
        if not (copied_func is None):
            copied_func(redf,wedf)
        wedf.close()
    return ch