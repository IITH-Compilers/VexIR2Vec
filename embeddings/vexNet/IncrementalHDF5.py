# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Efficiently appends preprocessed data into a resizable HDF5 file in chunks"""

import h5py
import pandas as pd
import numpy as np
import os


class incrementalHdf5:
    def __init__(
        self,
        file_path,
        target_size=1024 * 1024 * 30,
        preprocess_fn=None,
        wo=1,
        wt=1,
        wa=1,
    ):
        self.file_path = file_path
        self.target_size = target_size
        self.preprocess_fn = preprocess_fn
        self.current_size = 0
        self.index = 0
        self.h5_file = None
        self.datasets = {}
        self.wo = wo
        self.wt = wt
        self.wa = wa

    def createDataset(self, name, dtype, shape):

        # hf.create_dataset("strRefs", data=np.stack(df.strRefs.values))
        # hf.create_dataset("extlibs", data=np.stack(df.extlibs.values))
        # hf.create_dataset("embeddings", data=np.stack(df.embedding.values)
        self.datasets[name] = self.h5_file.createDataset(
            name, shape=shape, dtype=dtype, chunks=True, maxshape=(None, *shape[1:])
        )

    def appendData(self, data):
        for column, values in data.items():
            dataset = self.datasets[column]
            num_samples = len(values)
            if column != "key":
                values = np.stack(values)
            dataset.resize(self.index + num_samples, axis=0)
            dataset[self.index : self.index + num_samples] = values

        self.index += num_samples
        self.current_size += num_samples * values.itemsize

    def writeData(self, data):
        if data.empty:
            return

        if self.preprocess_fn is not None:
            data = self.preprocess_fn(data, self.wo, self.wt, self.wa)

        if self.h5_file is None:
            columns = list(data.columns)
            self.h5_file = h5py.File(self.file_path, "w")
            for column in columns:
                if column == "strRefs" or column == "extlibs":
                    dtype = np.float64
                    shape = (0, 100)
                elif column == "embedding":
                    dtype = np.float64
                    shape = (0, 100, 128)
                else:
                    dtype = data[column].dtype
                    shape = (0,)
                    # shape = data[column].shape
                self.createDataset(column, dtype, shape)

        self.appendData(data)

        if self.current_size >= self.target_size:
            self.current_size = 0
            self.h5_file.flush()

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
