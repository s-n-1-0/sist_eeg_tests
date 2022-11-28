# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import norm
path = "./dataset/lord2/ex.h5"
group_path = "annotations/Marker"
# %% データセット数

with h5py.File(path) as f:
    c = int(f[group_path].attrs["count"])
    print("Sum :" + str(c))
    labels = ["dark","light"]
    y = [f[f"{group_path}/{i}"].attrs["label"] for i in range(c)]
    plt.bar(labels,[y.count(l) for l in labels])
        
# %% ERP
r = 500
with h5py.File(path) as f:
    c = int(f[group_path].attrs["count"])
    darks = np.array([norm(f[f"{group_path}/{i}"][:,:r]) for i in range(c) if f[f"{group_path}/{i}"].attrs["label"] == "dark"])
    lights = np.array([norm(f[f"{group_path}/{i}"][:,:r]) for i in range(c) if f[f"{group_path}/{i}"].attrs["label"] == "light" ])
    erp_dark = (darks.sum(axis=0) / c)
    erp_light = (lights.sum(axis=0) / c)
    for i in range(10):
        plt.plot(range(r),erp_dark[i,:],label="dark")
        plt.plot(range(r),erp_light[i,:],label="light")
        plt.title(f"{i + 1}ch")
        plt.legend()
        plt.show()


# %% data[1] + erp
idx = 55
r = 500
with h5py.File(path) as f:
    c = int(f[group_path].attrs["count"])
    darks = np.array([norm(f[f"{group_path}/{i}"][:,:r]) for i in range(c) if f[f"{group_path}/{i}"].attrs["label"] == "dark"])
    lights = np.array([norm(f[f"{group_path}/{i}"][:,:r]) for i in range(c) if f[f"{group_path}/{i}"].attrs["label"] == "light" ])
    erp_dark = norm((darks[:,:,:].sum(axis=0) / c))
    erp_light = norm((lights[:,:,:].sum(axis=0) / c))
    merged_dark = norm((erp_dark + darks[idx,:,:]) / 2.0)
    merged_light = norm((erp_light + lights[idx,:,:]) / 2.0)
    for i in range(10):
        plt.plot(range(r),darks[idx,i,:],label="dark")
        plt.plot(range(r),erp_dark[i,:],label="erp_dark")
        plt.plot(range(r),merged_dark[i,:],label="merged_dark")
        plt.title(f"{i + 1}ch dark")
        plt.legend()
        plt.show()
        plt.plot(range(r),lights[idx,i,:],label="light")
        plt.plot(range(r),erp_light[i,:],label="erp_light")
        plt.plot(range(r),merged_light[i,:],label="merged_light")
        plt.title(f"{i + 1}ch light")
        plt.legend()
        plt.show()
# %%
