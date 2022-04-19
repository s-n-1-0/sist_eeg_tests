import pyedflib
def get_all_signals(edf:pyedflib.EdfReader):
    """
    edfファイルから全チャンネルの波形を取得して返します。
    """
    ewavs = []
    for idx,_ in enumerate(edf.getSignalLabels()):
        ewavs.append(edf.readSignal(idx))
    return ewavs