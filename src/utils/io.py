import h5py
import numpy as np

class HDF5Reader:
    def __init__(self, image_key="image", label_key="label"):
        self.image_key = image_key
        self.label_key = label_key

    def read(self, path):
        with h5py.File(path, "r") as f:
            return f[self.image_key][:].astype(np.float32), f[self.label_key][:].astype(np.float32)