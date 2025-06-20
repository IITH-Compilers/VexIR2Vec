# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.


"""Helper functions during inference"""

import os
import time
import torch
import random
import json
import numpy as np
import pandas as pd
import re


from utils import (
    printVex2vec,
    NUM_SB,
    INP_DIM,
    SEED,
    NUM_NEIGHBORS,
    STRIPPED_DIR,
    UNSTRIPPED_DIR,
    padAndSlice,
    strToMuldimNpArr,
    rwStrToArr,
    CNN_INP_DIM,
)
from argparse import ArgumentParser
from scipy.stats import rankdata
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


from scipy.spatial import cKDTree
from functools import partial
from multiprocessing import Pool, Queue
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    label_ranking_average_precision_score,
    average_precision_score,
    top_k_accuracy_score,
)


import warnings
import logging

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

device = torch.device("cpu")
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance()
scaler = preprocessing.StandardScaler()
print_logger = logging.getLogger("print_logger")


def modifyKey(x):

    x = re.split(", |\)", x)
    modified_str = x[0] + ")" + x[2]

    return modified_str


def getEmbedFromData(row):

    embed_O = (
        torch.from_numpy(rwStrToArr(row["embed_O"]))
        .view(-1, INP_DIM)
        .float()
        .to(device)
    )
    embed_T = (
        torch.from_numpy(rwStrToArr(row["embed_T"]))
        .view(-1, INP_DIM)
        .float()
        .to(device)
    )
    embed_A = (
        torch.from_numpy(rwStrToArr(row["embed_A"]))
        .view(-1, INP_DIM)
        .float()
        .to(device)
    )

    row["embed_O"] = embed_O
    row["embed_T"] = embed_T
    row["embed_A"] = embed_A

    strEmbed = np.fromstring(row["strRefs"].replace("[", "").replace("]", ""), sep=" ")

    row["strRefs"] = strEmbed

    strEmbed = torch.from_numpy(row["strRefs"])
    strEmbed = strEmbed.view(1, -1).float().to(device)
    row["strEmbed"] = strEmbed

    libEmbed = np.fromstring(row["extlibs"].replace("[", "").replace("]", ""), sep=" ")
    row["extlibs"] = libEmbed

    libEmbed = torch.from_numpy(row["extlibs"])
    libEmbed = libEmbed.view(1, -1).float().to(device)
    row["libEmbed"] = libEmbed

    if "cfg" in row:

        cfg = strToMuldimNpArr(row["cfg"])
        row["cfg"] = cfg

    return row


def getKeyEmbedList(unstrip_file, strip_file, cfg=False, func_addr=None):
    try:
        unstripped_df = pd.read_csv(
            unstrip_file,
            sep="\t",
            usecols=[0, 1, 2],
            header=None,
            names=["addr", "key", "name"],
        )
        if cfg == True:
            stripped_df = pd.read_csv(
                strip_file,
                sep="\t",
                usecols=[0, 1, 2, 3, 4, 5, 6],
                header=None,
                names=["addr", "key", "name", "strRefs", "extlibs", "embedding", "cfg"],
            )
        else:
            stripped_df = pd.read_csv(
                strip_file,
                sep="\t",
                usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                header=None,
                names=[
                    "addr",
                    "key",
                    "name",
                    "strRefs",
                    "extlibs",
                    "embed_O",
                    "embed_T",
                    "embed_A",
                ],
            )
        if func_addr is not None:
            unstripped_df = unstripped_df[unstripped_df["addr"] == func_addr]
            stripped_df = stripped_df[stripped_df["addr"] == func_addr]

        stripped_df.drop(["key"], axis=1, inplace=True)

        data = pd.merge(unstripped_df, stripped_df, on="addr", suffixes=("_x", "_y"))

        data = (
            data[data["name_x"].str.startswith("sub_") == False]
            .reset_index()
            .drop(["index"], axis=1)
            .astype({"addr": str})
        )
        data = (
            data[data["key"].str.contains("NO_FILE_SRC") == False]
            .reset_index()
            .drop(["index"], axis=1)
            .astype({"addr": str})
        )
        data = (
            data[
                data["name_y"].str.startswith("sub_")
                | data["name_y"].str.fullmatch("main")
            ]
            .reset_index()
            .drop(["index"], axis=1)
            .astype({"addr": str})
        )
        data["key"] = data["key"] + data["name_x"]
        data.key = data.key.apply(modifyKey)
        data = data.apply(getEmbedFromData, axis=1)
        addr_list = [int(addr) for addr in data["addr"].tolist()]
        key_list = data["key"].tolist()
        ### Skip for SAFE
        # emb_list = data['embedding'].tolist()
        emb_O_list = data["embed_O"].tolist()
        emb_T_list = data["embed_T"].tolist()
        emb_A_list = data["embed_A"].tolist()
        try:
            strEmb_list = data["strEmbed"].tolist()
        except Exception as e:
            with open("err_files.txt", "a") as f:
                f.write("strEmbed error: " + str(strip_file) + "\n")
            return []

        try:
            libEmb_list = data["libEmbed"].tolist()
        except Exception as e:
            with open("err_files.txt", "a") as f:
                f.write("libEmbed error: " + str(strip_file) + "\n")
            return []

        if cfg == True:
            cfg_list = data["cfg"].tolist()
            return list(
                zip(addr_list, key_list, emb_list, strEmb_list, libEmb_list, cfg_list)
            )
        else:
            return list(
                zip(
                    addr_list,
                    key_list,
                    emb_O_list,
                    emb_T_list,
                    emb_A_list,
                    strEmb_list,
                    libEmb_list,
                )
            )
    except Exception as e:
        return []


def getAddrEmbedList(unstrip_file, strip_file, cfg=False, func_addr=None):
    unstripped_df = pd.read_csv(
        unstrip_file,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        names=["addr", "key", "name"],
    )
    if cfg == True:
        stripped_df = pd.read_csv(
            strip_file,
            sep="\t",
            usecols=[0, 1, 2, 3, 4, 5, 6],
            header=None,
            names=["addr", "key", "name", "strRefs", "extlibs", "embedding", "cfg"],
        )
    else:
        cols_stripped = [
            "addr",
            "key",
            "name",
            "strRefs",
            "extlibs",
            "embed_O",
            "embed_T",
            "embed_A",
        ]
        stripped_df = pd.read_csv(
            strip_file,
            sep="\t",
            usecols=[0, 1, 2, 3, 4, 5, 6, 7],
            header=None,
            names=cols_stripped,
        )
    stripped_df.drop(["key"], axis=1, inplace=True)

    data = pd.merge(unstripped_df, stripped_df, on="addr", suffixes=("_x", "_y"))

    data = (
        data[data["name_x"].str.startswith("sub_") == False]
        .reset_index()
        .drop(["index"], axis=1)
        .astype({"addr": str})
    )
    data = (
        data[data["key"].str.contains("NO_FILE_SRC") == False]
        .reset_index()
        .drop(["index"], axis=1)
        .astype({"addr": str})
    )
    data = (
        data[
            data["name_y"].str.startswith("sub_") | data["name_y"].str.fullmatch("main")
        ]
        .reset_index()
        .drop(["index"], axis=1)
        .astype({"addr": str})
    )

    data = data.apply(getEmbedFromData, axis=1)

    addr_list = [int(func_addr) for func_addr in data["addr"].tolist()]
    emb_O_list = data["embed_O"].tolist()
    emb_T_list = data["embed_T"].tolist()
    emb_A_list = data["embed_A"].tolist()
    strEmb_list = data["strEmbed"].tolist()
    libEmb_list = data["libEmbed"].tolist()

    if cfg == True:
        cfg_list = data["cfg"].tolist()
        return list(
            zip(
                addr_list,
                emb_O_list,
                emb_T_list,
                emb_A_list,
                strEmb_list,
                libEmb_list,
                cfg_list,
            )
        )
    else:
        return list(
            zip(addr_list, emb_O_list, emb_T_list, emb_A_list, strEmb_list, libEmb_list)
        )


def validAddrList(data_path_x, pri_bin, model, device):

    validAddrList = []
    with open(os.path.join(data_path_x, pri_bin) + ".data", "r") as file:
        for line in file:

            if len(line.split("\t")) == 6:
                [func_addr, key, func_name, strRefs, extlibs, embedding] = line.split(
                    "\t"
                )
                if func_name.startswith("sub_") or func_name == "main":
                    try:
                        embedding = np.vstack(
                            [
                                np.fromstring(
                                    i.replace("[", "")
                                    .replace("]", "")
                                    .replace(" ", "  "),
                                    sep=" ",
                                )
                                for i in eval(embedding)
                            ]
                        )

                        embedding = torch.from_numpy(embedding)
                        embedding = (
                            embedding.view(NUM_SB, -1, INP_DIM).float().to(device)
                        )
                        strEmbed = torch.from_numpy(strRefs)
                        strEmbed = strEmbed.view(1, -1).float().to(device)

                        libEmbed = torch.from_numpy(extlibs)
                        libEmbed = libEmbed.view(1, -1).float().to(device)

                        with torch.no_grad():
                            output = model(embedding, strEmbed, libEmbed, test=True)
                            # print('output.shape: ', output.shape)
                        validAddrList.append((func_addr, output))

                    except Exception as err:
                        print(err)

    return validAddrList


def getLabelsAndSimScores(
    vexir2vec_func_list, vexir2vec_dist_list, test_func_list, labels, sim_scores
):
    tp_count = 0
    fp_count = 0

    for tuples, dists in zip(vexir2vec_func_list, vexir2vec_dist_list):

        assert len(tuples) == len(dists), "tuples len: {}, dists len: {}".format(
            len(tuples), len(dists)
        )
        counter = 0
        for idx in range(len(tuples)):
            # True positive
            if tuples[idx] in test_func_list:
                labels.append(1)
                tp_count += 1
                dist = dists[idx]
                sim_scores.append(1 / (1 + dist))
                break
            else:
                counter += 1
                # False positive
        if counter == len(tuples):
            fp_count += 1
            labels.append(0)
            dist = sum(dists) / len(dists)
            sim_scores.append(1 / (1 + dist))

    assert_count = fp_count + tp_count

    assert fp_count + tp_count == len(
        vexir2vec_func_list
    ), "fp_count: {}, tp_count: {}, len(vexir2vec_func_list): {}".format(
        fp_count, tp_count, len(vexir2vec_func_list)
    )

    return labels, sim_scores


class kdtreeNeigh:
    def __init__(self, tp_data):

        kdtree_fit_start = time.time()
        self.kdt = cKDTree(tp_data, leafsize=100)
        kdtree_fit_end = time.time()

    def getDuplicateDists(self, dist_arr):
        stride = 0
        while True:
            if NUM_NEIGHBORS + stride > dist_arr.shape[0]:
                stride -= NUM_NEIGHBORS - unq_cnt
                break
            r = np.array(dist_arr[: int(NUM_NEIGHBORS + stride)])
            r1 = np.unique(r)
            unq_cnt = r1.shape[0]
            if unq_cnt < NUM_NEIGHBORS:
                stride += NUM_NEIGHBORS - unq_cnt
            else:
                break
        return dist_arr[: int(NUM_NEIGHBORS + stride)]

    def getTopkNeighAndDist(self, tp_query, k, val=False):
        if val:
            NUM_NEIGHBORS_LOCAL = 30
        else:
            NUM_NEIGHBORS_LOCAL = NUM_NEIGHBORS

        kdtree_query_start = time.time()
        vexir2vec_dist, vexir2vec_index = self.kdt.query(tp_query, k=k, p=2, workers=-1)
        kdtree_query_end = time.time()

        vexir2vec_dist = np.atleast_2d(vexir2vec_dist)
        vexir2vec_index = np.atleast_2d(vexir2vec_index)

        duplicate_neigh_start = time.time()
        duplicate_noise_start = time.time()
        stride = np.zeros(vexir2vec_dist.shape[0])

        unq_cnt = 0
        duplicate_apply_start = time.time()
        actual_k = min(NUM_NEIGHBORS_LOCAL, vexir2vec_dist.shape[1])
        for i in range(0, vexir2vec_dist.shape[0]):
            while True:
                if actual_k + stride[i] > vexir2vec_dist.shape[1]:
                    stride[i] -= actual_k - unq_cnt
                    break
                r = np.array(vexir2vec_dist[i][: int(actual_k + stride[i])])
                r1 = np.unique(r)
                unq_cnt = r1.shape[0]
                if unq_cnt < actual_k:
                    stride[i] += actual_k - unq_cnt
                else:
                    break
        vexir2vec_dist1 = []
        actual_k = min(NUM_NEIGHBORS_LOCAL, vexir2vec_dist.shape[1])
        for i in range(0, vexir2vec_dist.shape[0]):
            vexir2vec_dist1.append(vexir2vec_dist[i][: int(actual_k + stride[i])])
        vexir2vec_dist1 = np.array(vexir2vec_dist1)

        duplicate_apply_end = time.time()

        duplicate_index_start = time.time()
        vexir2vec_index1 = []

        for i in range(0, vexir2vec_index.shape[0]):
            vexir2vec_index1.append(
                vexir2vec_index[i][: int(vexir2vec_dist1[i].shape[0])]
            )
        vexir2vec_index1 = np.array(vexir2vec_index1)
        duplicate_index_end = time.time()

        duplicate_neigh_end = time.time()

        return vexir2vec_index1, vexir2vec_dist1


def isKeyMatch(arr, dist_arr, search_key_list, test_key_list):
    index = arr[-1]
    dists = dist_arr[index]
    print(arr.shape)
    arr = arr[:-1]
    if arr.shape[0] == 1:
        arr = arr[0]
    print(arr)
    keymatch_arr = search_key_list[arr] == test_key_list[index]
    print(keymatch_arr)
    return keymatch_arr


def loadSearchData(search_config, x86_data_path, arm_data_path, use_cfg):
    test_dict = {
        "coreutils": [
            "who",
            "stat",
            "tee",
            "sha256sum",
            "sha384sum",
            "sha224sum",
            "base32",
            "sha512sum",
            "unexpand",
            "expand",
            "base64",
            "chroot",
            "md5sum",
            "env",
            "sha1sum",
            "uniq",
            "readlink",
            "fmt",
            "stty",
            "cksum",
            "head",
            "realpath",
            "uptime",
            "wc",
            "b2sum",
            "tr",
            "join",
            "numfmt",
            "factor",
            "split",
            "dd",
            "rm",
            "shred",
            "touch",
        ],
        "diffutils": ["cmp", "sdiff"],
        "findutils": ["xargs"],
        "gzip": ["gzip"],
        "lua": ["lua", "luac"],
        "curl": ["curl"],
        "putty": [
            "cgtest",
            "fuzzterm",
            "osxlaunch",
            "plink",
            "pscp",
            "psftp",
            "psocks",
            "psusan",
            "puttygen",
            "testcrypt",
            "testsc",
            "testzlib",
            "uppity",
        ],
    }
    search_config_key_embed_list = []

    for proj in test_dict.keys():

        if "x86" in search_config:
            search_config_data_path = os.path.join(
                x86_data_path, proj, search_config, STRIPPED_DIR
            )
        elif "arm" in search_config:
            search_config_data_path = os.path.join(
                arm_data_path, proj, search_config, STRIPPED_DIR
            )

        for bin_name in test_dict[proj]:
            search_data_list = []

            stripped_data = os.path.join(search_config_data_path, bin_name) + ".data"
            unstripped_data = (
                os.path.join(
                    search_config_data_path.replace(STRIPPED_DIR, UNSTRIPPED_DIR),
                    bin_name,
                )
                + ".data"
            )
            if not os.path.exists(stripped_data) or not os.path.exists(unstripped_data):
                continue
            with open(stripped_data, "r") as strip_file, open(
                unstripped_data, "r"
            ) as unstrip_file:
                search_data_list = getKeyEmbedList(unstrip_file, strip_file, use_cfg)
                for search_data in search_data_list:
                    if use_cfg:
                        search_config_key_embed_list.append(
                            (
                                search_config + proj + bin_name,
                                search_data[0],
                                search_data[1],
                                search_data[2],
                                search_data[3],
                                search_data[4],
                                search_data[5],
                            )
                        )
                    else:
                        search_config_key_embed_list.append(
                            (
                                search_config + proj + bin_name,
                                search_data[0],
                                search_data[1],
                                search_data[2],
                                search_data[3],
                                search_data[4],
                            )
                        )
    return search_config_key_embed_list


def newGetKeyEmbedList(strip_file):

    stripped_df = pd.read_csv(
        strip_file,
        sep="\t",
        usecols=[0, 1, 2, 3, 4, 5],
        header=None,
        names=["addr", "key", "name", "strRefs", "extlibs", "embedding"],
    )

    data = stripped_df

    data = data.apply(getEmbedFromData, axis=1)

    emb_list = data["embedding"].tolist()
    # try:
    strEmb_list = data["strEmbed"].tolist()

    libEmb_list = data["libEmbed"].tolist()

    return list(zip(emb_list, strEmb_list, libEmb_list))
