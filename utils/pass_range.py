import numpy as np
def pass_range(array:np.ndarray,min_val:float = None,max_val:float = None):
    """指定した範囲を満たすインデックスを返します。

    Args:
        array (np.ndarray): 指定配列
        min_val (float, optional): 指定配列から切り取る範囲の最小値
        max_val (float, optional): 指定配列から切り取る範囲の最大値

    Returns:
        tuple: (範囲インデックス配列,範囲指定で切り取られたarray)
    """
    if min_val is None:
        min_val = array[0]
    if max_val is None:
        max_val = array[-1]
    pass_range_indexes = np.where((array >= min_val) & (array <= max_val))[0]
    pass_array = array[pass_range_indexes]
    return (pass_range_indexes,pass_array)