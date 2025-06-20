# coding:utf-8
import os
import ctypes
import numpy as np


class TrainDataSampler(object):
    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TrainDataLoader(object):
    def __init__(
        self,
        in_path="./",
        tri_file=None,
        ent_file=None,
        rel_file=None,
        batch_size=None,
        nbatches=None,
        threads=8,
        sampling_mode="normal",
        bern_flag=False,
        filter_flag=True,
        neg_ent=1,
        neg_rel=0,
    ):

        base_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../release/Base.so")
        )
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.in_path = in_path
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        print("--------------------------------------------------------")
        print("self.in_path == ", self.in_path)
        print("--------------------------------------------------------")
        if in_path != None:
            self.tri_file = in_path + "train2id.txt"
            self.ent_file = in_path + "entity2id.txt"
            self.rel_file = in_path + "relation2id.txt"
        """set essential parameters"""
        self.work_threads = threads
        self.nbatches = nbatches
        self.batch_size = batch_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        self.cross_sampling_flag = 0
        self.read()

    def read(self):
        if self.in_path != None:
            self.lib.setInPath(
                ctypes.create_string_buffer(
                    self.in_path.encode(), len(self.in_path) * 2
                )
            )
        else:
            print("here2")
            self.lib.setTrainPath(
                ctypes.create_string_buffer(
                    self.tri_file.encode(), len(self.tri_file) * 2
                )
            )
            self.lib.setEntPath(
                ctypes.create_string_buffer(
                    self.ent_file.encode(), len(self.ent_file) * 2
                )
            )
            self.lib.setRelPath(
                ctypes.create_string_buffer(
                    self.rel_file.encode(), len(self.rel_file) * 2
                )
            )

        self.lib.setTrainPath(
            ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2)
        )
        self.lib.setEntPath(
            ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2)
        )
        self.lib.setRelPath(
            ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2)
        )

        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.tripleTotal = self.lib.getTrainTotal()
        print("triple total = " + str(self.tripleTotal))

        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size
        self.batch_seq_size = self.batch_size * (
            1 + self.negative_ent + self.negative_rel
        )

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0,
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t,
            "batch_r": self.batch_r,
            "batch_y": self.batch_y,
            "mode": "normal",
        }

    def samplingHead(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            -1,
            self.filter,
            0,
            0,
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t[: self.batch_size],
            "batch_r": self.batch_r[: self.batch_size],
            "batch_y": self.batch_y,
            "mode": "head_batch",
        }

    def samplingTail(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            1,
            self.filter,
            0,
            0,
        )
        return {
            "batch_h": self.batch_h[: self.batch_size],
            "batch_t": self.batch_t,
            "batch_r": self.batch_r[: self.batch_size],
            "batch_y": self.batch_y,
            "mode": "tail_batch",
        }

    def crossSampling(self):
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            return self.samplingHead()
        else:
            return self.samplingTail()

    """interfaces to set essential parameters"""

    def setWorkThreads(self, work_threads):
        self.work_threads = work_threads

    def setInPath(self, in_path):
        self.in_path = in_path

    def setNbatches(self, nbatches):
        self.nbatches = nbatches

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def setEntNegRate(self, rate):
        self.negative_ent = rate

    def setRelNegRate(self, rate):
        self.negative_rel = rate

    def setBernFlag(self, bern):
        self.bern = bern

    def setFilterFlag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def getBatchSize(self):
        return self.batch_size

    def getEntTot(self):
        return self.entTotal

    def getRelTot(self):
        return self.relTotal

    def getTripleTot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.crossSampling)

    def __len__(self):
        return self.nbatches
