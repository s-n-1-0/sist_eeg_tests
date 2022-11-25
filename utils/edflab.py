# %%
import pandas as pd
# %% CSVとEDFファイルのペア情報を取得する
def get_edfcsv_filenames(path:str,encoding:str="ansi"):
    df = pd.read_csv(path,encoding=encoding)
    return df[["EDF","CSV"]].values
