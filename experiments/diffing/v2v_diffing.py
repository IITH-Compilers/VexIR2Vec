# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Diffing Script"""

# To use this script as standalone for inferencing particular set:
# Usage: python v2v_diffing.py -bmp /path/to/model -dp /path/to/project-wise-data-files/ -test_csv /path/to/project-config-wise-groundtruth.csv -out_dir /path/to/results -res_dir /path/to/results
import sys
import os

sys.path.append(os.path.abspath("../../embeddings/vexNet"))
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
from utils_inference import kdtreeNeigh, getAddrEmbedList, getLabelsAndSimScores
from sklearn.metrics.pairwise import cosine_similarity


import warnings

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

device = torch.device("cuda")
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance()
scaler = preprocessing.StandardScaler()


class cosineSimilarityNeigh:
    def __init__(self, data):
        self.data = data

    def getTopkNeighAndDist(self, query, k):
        k = NUM_NEIGHBORS
        cosine_sim = cosine_similarity(query, self.data)
        topk_indices = np.argsort(cosine_sim, axis=1)[:, -k:][:, ::-1]
        topk_distances = (
            1 - cosine_sim[np.arange(len(cosine_sim))[:, None], topk_indices]
        )
        return topk_indices, topk_distances

        # This function is used to calculate scores in validation


def getScores(model, data_path, use_cfg, csv_test, cosine_similarity=False):

    exp_dir = os.path.basename(os.path.dirname(csv_test))
    if "arm" in exp_dir and "x86" in exp_dir:
        src_data_path = data_path
        tgt_data_path = data_path.replace("x86", "arm")
    elif "x86" in exp_dir:
        src_data_path = data_path
        tgt_data_path = data_path
    else:
        src_data_path = data_path.replace("x86", "arm")
        tgt_data_path = data_path.replace("x86", "arm")

    base = os.path.basename(csv_test)
    filename_tokens = base.split("-full-")
    config = filename_tokens[1].split(".csv")[0]
    proj = filename_tokens[0]

    sub = config.rsplit("-", 4)
    config_x = sub[0]
    config_y = "-".join(sub[1:])
    suffix_x = "_" + config_x
    suffix_y = "_" + config_y

    data_path_x = os.path.join(src_data_path, config_x, STRIPPED_DIR)
    data_path_y = os.path.join(tgt_data_path, config_y, STRIPPED_DIR)
    tp_data = pd.read_csv(csv_test)

    bin_dict = {}
    for _, test_row in tp_data.iterrows():
        pri = test_row["bin_name" + suffix_x]
        sec = test_row["bin_name" + suffix_y]

        bin_dict.setdefault(pri, set()).add(sec)

    tp, src_tp, tgt_tp = 0, 0, 0
    fp, src_fp, tgt_fp = 0, 0, 0
    for pri_bin in bin_dict.keys():
        for sec_bin in bin_dict[pri_bin]:
            vexir2vec_func_list_src = []
            vexir2vec_dist_list_src = []
            vexir2vec_func_list_tgt = []
            vexir2vec_dist_list_tgt = []
            test_func_list_src = []
            test_func_list_tgt = []
            pri_addr_list = []
            sec_addr_list = []
            output_pri_list = []
            output_sec_list = []

            t = time.time()
            with open(
                os.path.join(data_path_x, pri_bin) + ".data", "r"
            ) as strip_file, open(
                os.path.join(data_path_x.replace(STRIPPED_DIR, UNSTRIPPED_DIR), pri_bin)
                + ".data",
                "r",
            ) as unstrip_file:
                pri_data_list = getAddrEmbedList(unstrip_file, strip_file, use_cfg)

            opcEmbed_list_pri = list(map(lambda x: x[1], pri_data_list))
            opcEmbed_pri = torch.squeeze(torch.stack(opcEmbed_list_pri, dim=0), 1)

            tyEmbed_list_pri = list(map(lambda x: x[2], pri_data_list))
            tyEmbed_pri = torch.squeeze(torch.stack(tyEmbed_list_pri, dim=0))

            argEmbed_list_pri = list(map(lambda x: x[3], pri_data_list))
            argEmbed_pri = torch.squeeze(torch.stack(argEmbed_list_pri, dim=0))

            strEmbed_list_pri = list(map(lambda x: x[4], pri_data_list))
            strEmbed_pri = torch.squeeze(torch.stack(strEmbed_list_pri, dim=0))

            libEmbed_list_pri = list(map(lambda x: x[5], pri_data_list))
            libEmbed_pri = torch.squeeze(torch.stack(libEmbed_list_pri, dim=0))

            pri_addr_list = list(map(lambda x: x[0], pri_data_list))

            dataset = TensorDataset(
                opcEmbed_pri, tyEmbed_pri, argEmbed_pri, strEmbed_pri, libEmbed_pri
            )
            test_dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=20
            )
            # print(len(test_dataloader))

            for test_data in test_dataloader:
                with torch.no_grad():

                    opc_emb = test_data[0].to(torch.device("cuda"))
                    ty_emb = test_data[1].to(torch.device("cuda"))
                    arg_emb = test_data[2].to(torch.device("cuda"))
                    str_emb = test_data[3].to(torch.device("cuda"))
                    lib_emb = test_data[4].to(torch.device("cuda"))
                    res, _ = model(
                        opc_emb, ty_emb, arg_emb, str_emb, lib_emb, test=True
                    )
                    output_pri_list.extend(res)

            pri_addr_list = list(zip(pri_addr_list, output_pri_list))

            with open(
                os.path.join(data_path_y, sec_bin) + ".data", "r"
            ) as strip_file, open(
                os.path.join(data_path_y.replace(STRIPPED_DIR, UNSTRIPPED_DIR), sec_bin)
                + ".data",
                "r",
            ) as unstrip_file:
                sec_data_list = getAddrEmbedList(unstrip_file, strip_file, use_cfg)

            opcEmbed_list_sec = list(map(lambda x: x[1], sec_data_list))
            opcEmbed_sec = torch.squeeze(torch.stack(opcEmbed_list_sec, dim=0), 1)

            tyEmbed_list_sec = list(map(lambda x: x[2], sec_data_list))
            tyEmbed_sec = torch.squeeze(torch.stack(tyEmbed_list_sec, dim=0))

            argEmbed_list_sec = list(map(lambda x: x[3], sec_data_list))
            argEmbed_sec = torch.squeeze(torch.stack(argEmbed_list_sec, dim=0))

            strEmbed_list_sec = list(map(lambda x: x[4], sec_data_list))
            strEmbed_sec = torch.squeeze(torch.stack(strEmbed_list_sec, dim=0))

            libEmbed_list_sec = list(map(lambda x: x[5], sec_data_list))
            libEmbed_sec = torch.squeeze(torch.stack(libEmbed_list_sec, dim=0))

            sec_addr_list = list(map(lambda x: x[0], sec_data_list))

            dataset = TensorDataset(
                opcEmbed_sec, tyEmbed_sec, argEmbed_sec, strEmbed_sec, libEmbed_sec
            )
            test_dataloader = DataLoader(dataset, shuffle=False, num_workers=20)
            for test_data in test_dataloader:
                with torch.no_grad():

                    opc_emb = test_data[0].to(torch.device("cuda"))
                    ty_emb = test_data[1].to(torch.device("cuda"))
                    arg_emb = test_data[2].to(torch.device("cuda"))
                    str_emb = test_data[3].to(torch.device("cuda"))
                    lib_emb = test_data[4].to(torch.device("cuda"))

                    res1, _ = model(
                        opc_emb, ty_emb, arg_emb, str_emb, lib_emb, test=True
                    )
                    output_sec_list.extend(res1)
            sec_addr_list = list(zip(sec_addr_list, output_sec_list))

            tp_src = np.array(
                [torch.squeeze(tup[1]).cuda().cpu().numpy() for tup in pri_addr_list]
            ).astype(np.float32)

            tp_tgt = np.array(
                [torch.squeeze(tup[1]).cuda().cpu().numpy() for tup in sec_addr_list]
            ).astype(np.float32)

            pri_addr_list = [int(i[0]) for i in pri_addr_list]
            sec_addr_list = [int(i[0]) for i in sec_addr_list]

            if cosine_similarity:
                cosine_neigh = cosineSimilarityNeigh(tp_tgt)
                (
                    vexir2vec_func_index_src,
                    vexir2vec_func_dist_src,
                ) = cosine_neigh.getTopkNeighAndDist(tp_src, len(tp_tgt))
            else:
                kdt = kdtreeNeigh(tp_tgt)
                (
                    vexir2vec_func_index_src,
                    vexir2vec_func_dist_src,
                ) = kdt.getTopkNeighAndDist(tp_src, len(tp_tgt))

            if cosine_similarity:
                cosine_neigh = cosineSimilarityNeigh(tp_src)
                (
                    vexir2vec_func_index_tgt,
                    vexir2vec_func_dist_tgt,
                ) = cosine_neigh.getTopkNeighAndDist(tp_tgt, len(tp_src))
            else:
                kdt = kdtreeNeigh(tp_src)
                (
                    vexir2vec_func_index_tgt,
                    vexir2vec_func_dist_tgt,
                ) = kdt.getTopkNeighAndDist(tp_tgt, len(tp_src))

            bin_data = (
                tp_data[
                    (tp_data["bin_name" + suffix_x] == pri_bin)
                    & (tp_data["bin_name" + suffix_y] == sec_bin)
                ]
                .reset_index()
                .drop(["index"], axis=1)
            )
            address_x_list = [
                int(func_addr) for func_addr in bin_data["address" + suffix_x].tolist()
            ]
            address_y_list = [
                int(func_addr) for func_addr in bin_data["address" + suffix_y].tolist()
            ]
            test_func_list_src = list(zip(address_x_list, address_y_list))
            test_func_list_tgt = list(zip(address_y_list, address_x_list))

            for index in range(len(vexir2vec_func_index_src)):
                tuples = [
                    (int(pri_addr_list[index]), int(sec_addr_list[tgt_index]))
                    for tgt_index in vexir2vec_func_index_src[index]
                ]
                dists = [
                    float(vexir2vec_func_dist_src[index][sub_index])
                    for sub_index in range(len(vexir2vec_func_index_src[index]))
                ]
                vexir2vec_func_list_src.append(tuples)
                vexir2vec_dist_list_src.append(dists)

            for index in range(len(vexir2vec_func_index_tgt)):
                tuples = [
                    (int(sec_addr_list[index]), int(pri_addr_list[tgt_index]))
                    for tgt_index in vexir2vec_func_index_tgt[index]
                ]
                dists = [
                    float(vexir2vec_func_dist_tgt[index][sub_index])
                    for sub_index in range(len(vexir2vec_func_index_tgt[index]))
                ]
                vexir2vec_func_list_tgt.append(tuples)
                vexir2vec_dist_list_tgt.append(dists)

                # True Positive
            src_tp += sum(
                any(tup in tuples for tuples in vexir2vec_func_list_src)
                for tup in test_func_list_src
            )
            tgt_tp += sum(
                any(tup in tuples for tuples in vexir2vec_func_list_tgt)
                for tup in test_func_list_tgt
            )

            # False Positive
            src_fp += sum(
                all(tup not in test_func_list_src for tup in tuples)
                for tuples in vexir2vec_func_list_src
            )
            tgt_fp += sum(
                all(tup not in test_func_list_tgt for tup in tuples)
                for tuples in vexir2vec_func_list_tgt
            )

    tp = (src_tp + tgt_tp) // 2
    fp = (src_fp + tgt_fp) // 2
    fn = tp_data.shape[0] - tp

    if tp != 0 or fp != 0:
        prec = round(tp / (tp + fp), 3)
    else:
        prec = "NA"

    if tp != 0 or fn != 0:
        rec = round(tp / (tp + fn), 3)
    else:
        rec = "NA"

    if prec != "NA" and rec != "NA" and (prec != 0 or rec != 0):
        f1 = round((2 * prec * rec) / (prec + rec), 3)
    else:
        f1 = "NA"

    return prec, rec, f1


def getTopkScores(args, csv_test):
    # args, exp_dir, csv_test = triple
    exp_dir = os.path.basename(os.path.dirname(csv_test))
    name = os.path.basename(csv_test).split(".")[0]
    if not os.path.exists(f"{args .res_dir }/{exp_dir }"):
        os.makedirs(f"{args .res_dir }/{exp_dir }", exist_ok=True)
    res_file = f"{args .res_dir }/{exp_dir }/{name }.results"
    if os.path.isfile(res_file):
        print(f"{res_file } exists.")
        return

    model = torch.load(args.best_model_path, map_location=device)

    model.eval()

    data_path = args.data_path
    if "arm" in exp_dir and "x86" in exp_dir:
        src_data_path = data_path
        tgt_data_path = data_path.replace("x86", "arm")
    elif "x86" in exp_dir:
        src_data_path = data_path
        tgt_data_path = data_path
    else:
        src_data_path = data_path.replace("x86", "arm")
        tgt_data_path = data_path.replace("x86", "arm")

    base = os.path.basename(csv_test)
    filename_tokens = base.split("-full-")
    config = filename_tokens[1].split(".csv")[0]
    proj = filename_tokens[0]

    sub = config.rsplit("-", 4)
    config_x = sub[0]
    config_y = "-".join(sub[1:])
    suffix_x = "_" + config_x
    suffix_y = "_" + config_y

    data_path_x = os.path.join(src_data_path, config_x, STRIPPED_DIR)
    data_path_y = os.path.join(tgt_data_path, config_y, STRIPPED_DIR)
    tp_data = pd.read_csv(csv_test)
    # print(tp_data.columns)

    bin_dict = {}
    sim_scores_src_tmp = []
    labels_src_tmp = []
    sim_scores_tgt_tmp = []
    labels_tgt_tmp = []

    for _, test_row in tp_data.iterrows():
        pri = test_row["bin_name" + suffix_x]
        sec = test_row["bin_name" + suffix_y]

        bin_dict.setdefault(pri, set()).add(sec)

    tp, src_tp, tgt_tp = 0, 0, 0
    fp, src_fp, tgt_fp = 0, 0, 0
    for pri_bin in bin_dict.keys():
        for sec_bin in bin_dict[pri_bin]:
            vexir2vec_func_list_src = []
            vexir2vec_dist_list_src = []
            vexir2vec_func_list_tgt = []
            vexir2vec_dist_list_tgt = []
            test_func_list_src = []
            test_func_list_tgt = []
            pri_addr_list = []
            sec_addr_list = []
            output_pri_list = []
            output_sec_list = []

            t = time.time()
            with open(
                os.path.join(data_path_x, pri_bin) + ".data", "r"
            ) as strip_file, open(
                os.path.join(data_path_x.replace(STRIPPED_DIR, UNSTRIPPED_DIR), pri_bin)
                + ".data",
                "r",
            ) as unstrip_file:
                pri_data_list = getAddrEmbedList(unstrip_file, strip_file, args.use_cfg)

            opcEmbed_list_pri = list(map(lambda x: x[1], pri_data_list))
            opcEmbed_pri = torch.squeeze(torch.stack(opcEmbed_list_pri, dim=0), 1)

            tyEmbed_list_pri = list(map(lambda x: x[2], pri_data_list))
            tyEmbed_pri = torch.squeeze(torch.stack(tyEmbed_list_pri, dim=0))

            argEmbed_list_pri = list(map(lambda x: x[3], pri_data_list))
            argEmbed_pri = torch.squeeze(torch.stack(argEmbed_list_pri, dim=0))

            strEmbed_list_pri = list(map(lambda x: x[4], pri_data_list))
            strEmbed_pri = torch.squeeze(torch.stack(strEmbed_list_pri, dim=0))

            libEmbed_list_pri = list(map(lambda x: x[5], pri_data_list))
            libEmbed_pri = torch.squeeze(torch.stack(libEmbed_list_pri, dim=0))

            pri_addr_list = list(map(lambda x: x[0], pri_data_list))

            dataset = TensorDataset(
                opcEmbed_pri, tyEmbed_pri, argEmbed_pri, strEmbed_pri, libEmbed_pri
            )
            test_dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=20
            )
            for test_data in test_dataloader:
                with torch.no_grad():

                    opc_emb = test_data[0].to(torch.device("cuda"))
                    ty_emb = test_data[1].to(torch.device("cuda"))
                    arg_emb = test_data[2].to(torch.device("cuda"))
                    str_emb = test_data[3].to(torch.device("cuda"))
                    lib_emb = test_data[4].to(torch.device("cuda"))
                    res, _ = model(
                        opc_emb, ty_emb, arg_emb, str_emb, lib_emb, test=True
                    )

                    output_pri_list.extend(res)

            pri_addr_list = list(zip(pri_addr_list, output_pri_list))

            with open(
                os.path.join(data_path_y, sec_bin) + ".data", "r"
            ) as strip_file, open(
                os.path.join(data_path_y.replace(STRIPPED_DIR, UNSTRIPPED_DIR), sec_bin)
                + ".data",
                "r",
            ) as unstrip_file:
                sec_data_list = getAddrEmbedList(unstrip_file, strip_file, args.use_cfg)

            opcEmbed_list_sec = list(map(lambda x: x[1], sec_data_list))
            opcEmbed_sec = torch.squeeze(torch.stack(opcEmbed_list_sec, dim=0), 1)

            tyEmbed_list_sec = list(map(lambda x: x[2], sec_data_list))
            tyEmbed_sec = torch.squeeze(torch.stack(tyEmbed_list_sec, dim=0))

            argEmbed_list_sec = list(map(lambda x: x[3], sec_data_list))
            argEmbed_sec = torch.squeeze(torch.stack(argEmbed_list_sec, dim=0))

            strEmbed_list_sec = list(map(lambda x: x[4], sec_data_list))
            strEmbed_sec = torch.squeeze(torch.stack(strEmbed_list_sec, dim=0))

            libEmbed_list_sec = list(map(lambda x: x[5], sec_data_list))
            libEmbed_sec = torch.squeeze(torch.stack(libEmbed_list_sec, dim=0))

            sec_addr_list = list(map(lambda x: x[0], sec_data_list))

            dataset = TensorDataset(
                opcEmbed_sec, tyEmbed_sec, argEmbed_sec, strEmbed_sec, libEmbed_sec
            )
            test_dataloader = DataLoader(dataset, shuffle=False, num_workers=20)
            for test_data in test_dataloader:
                with torch.no_grad():

                    opc_emb = test_data[0].to(torch.device("cuda"))
                    ty_emb = test_data[1].to(torch.device("cuda"))
                    arg_emb = test_data[2].to(torch.device("cuda"))
                    str_emb = test_data[3].to(torch.device("cuda"))
                    lib_emb = test_data[4].to(torch.device("cuda"))
                    res1, _ = model(
                        opc_emb, ty_emb, arg_emb, str_emb, lib_emb, test=True
                    )

                    output_sec_list.extend(res1)

            sec_addr_list = list(zip(sec_addr_list, output_sec_list))

            tp_src = np.array(
                [torch.squeeze(tup[1]).cuda().cpu().numpy() for tup in pri_addr_list]
            ).astype(np.float32)
            tp_tgt = np.array(
                [torch.squeeze(tup[1]).cuda().cpu().numpy() for tup in sec_addr_list]
            ).astype(np.float32)

            pri_addr_list = [int(i[0]) for i in pri_addr_list]
            sec_addr_list = [int(i[0]) for i in sec_addr_list]

            print("tp_src shape: ", tp_src.shape)
            print("tp_tgt shape: ", tp_tgt.shape)

            if args.cosine_similarity:
                cosine_neigh = cosineSimilarityNeigh(tp_tgt)
                (
                    vexir2vec_func_index_src,
                    vexir2vec_func_dist_src,
                ) = cosine_neigh.getTopkNeighAndDist(tp_src, len(tp_tgt))
            else:
                kdt = kdtreeNeigh(tp_tgt)
                (
                    vexir2vec_func_index_src,
                    vexir2vec_func_dist_src,
                ) = kdt.getTopkNeighAndDist(tp_src, len(tp_tgt))

            print("vexir2vec_func_dist_src shape: ", vexir2vec_func_dist_src[5].shape)

            if args.cosine_similarity:
                cosine_neigh = cosineSimilarityNeigh(tp_src)
                (
                    vexir2vec_func_index_tgt,
                    vexir2vec_func_dist_tgt,
                ) = cosine_neigh.getTopkNeighAndDist(tp_tgt, len(tp_src))
            else:
                kdt = kdtreeNeigh(tp_src)
                (
                    vexir2vec_func_index_tgt,
                    vexir2vec_func_dist_tgt,
                ) = kdt.getTopkNeighAndDist(tp_tgt, len(tp_src))

            print("vexir2vec_func_dist_tgt shape: ", vexir2vec_func_dist_tgt[5].shape)

            bin_data = (
                tp_data[
                    (tp_data["bin_name" + suffix_x] == pri_bin)
                    & (tp_data["bin_name" + suffix_y] == sec_bin)
                ]
                .reset_index()
                .drop(["index"], axis=1)
            )
            address_x_list = [
                int(func_addr) for func_addr in bin_data["address" + suffix_x].tolist()
            ]
            address_y_list = [
                int(func_addr) for func_addr in bin_data["address" + suffix_y].tolist()
            ]
            test_func_list_src = list(zip(address_x_list, address_y_list))
            test_func_list_tgt = list(zip(address_y_list, address_x_list))

            for index in range(len(vexir2vec_func_index_src)):
                tuples = [
                    (int(pri_addr_list[index]), int(sec_addr_list[tgt_index]))
                    for tgt_index in vexir2vec_func_index_src[index]
                ]
                dists = [
                    float(vexir2vec_func_dist_src[index][sub_index])
                    for sub_index in range(len(vexir2vec_func_index_src[index]))
                ]
                vexir2vec_func_list_src.append(tuples)
                vexir2vec_dist_list_src.append(dists)

            for index in range(len(vexir2vec_func_index_tgt)):
                tuples = [
                    (int(sec_addr_list[index]), int(pri_addr_list[tgt_index]))
                    for tgt_index in vexir2vec_func_index_tgt[index]
                ]
                dists = [
                    float(vexir2vec_func_dist_tgt[index][sub_index])
                    for sub_index in range(len(vexir2vec_func_index_tgt[index]))
                ]
                vexir2vec_func_list_tgt.append(tuples)
                vexir2vec_dist_list_tgt.append(dists)

                # True Positive
            src_tp += sum(
                any(tup in tuples for tuples in vexir2vec_func_list_src)
                for tup in test_func_list_src
            )
            tgt_tp += sum(
                any(tup in tuples for tuples in vexir2vec_func_list_tgt)
                for tup in test_func_list_tgt
            )

            # False Positive
            src_fp += sum(
                all(tup not in test_func_list_src for tup in tuples)
                for tuples in vexir2vec_func_list_src
            )
            tgt_fp += sum(
                all(tup not in test_func_list_tgt for tup in tuples)
                for tuples in vexir2vec_func_list_tgt
            )

            labels_src_tmp, sim_scores_src_tmp = getLabelsAndSimScores(
                vexir2vec_func_list_src,
                vexir2vec_dist_list_src,
                test_func_list_src,
                labels_src_tmp,
                sim_scores_src_tmp,
            )
            labels_tgt_tmp, sim_scores_tgt_tmp = getLabelsAndSimScores(
                vexir2vec_func_list_tgt,
                vexir2vec_dist_list_tgt,
                test_func_list_tgt,
                labels_tgt_tmp,
                sim_scores_tgt_tmp,
            )

    tp = (src_tp + tgt_tp) // 2
    fp = (src_fp + tgt_fp) // 2
    fn = tp_data.shape[0] - tp

    if tp != 0 or fp != 0:
        prec = round(tp / (tp + fp), 3)
    else:
        prec = "NA"

    if tp != 0 or fn != 0:
        rec = round(tp / (tp + fn), 3)
    else:
        rec = "NA"

    if prec != "NA" and rec != "NA" and (prec != 0 or rec != 0):
        f1 = round((2 * prec * rec) / (prec + rec), 3)
    else:
        f1 = "NA"

        # prec, rec, f1 = check_and_calcPRFScore(tp, fp, fn)
    resultStr = f"{base .upper ()}\n{tp }\n{fp }\n{fn }\n{prec }\n{rec }\n{f1 }\n"
    # print(resultStr)
    with open(res_file, "w") as ofile:
        ofile.write(resultStr)

    sim_scores_src = {}
    labels_src = {}
    if config not in sim_scores_src.keys():
        sim_scores_src[config] = {}
    if config not in labels_src.keys():
        labels_src[config] = {}

    sim_scores_tgt = {}
    labels_tgt = {}
    if config not in sim_scores_tgt.keys():
        sim_scores_tgt[config] = {}
    if config not in labels_tgt.keys():
        labels_tgt[config] = {}

    print("sim_scores_src_tmp:", len(sim_scores_src_tmp))
    print("labels_src_tmp:", len(labels_src_tmp))
    print("sim_scores_tgt_tmp:", len(sim_scores_tgt_tmp))
    print("labels_tgt_tmp:", len(labels_tgt_tmp))

    if proj not in sim_scores_src[config].keys():
        sim_scores_src[config][proj] = sim_scores_src_tmp
    if proj not in labels_src[config].keys():
        labels_src[config][proj] = labels_src_tmp

    if proj not in sim_scores_tgt[config].keys():
        sim_scores_tgt[config][proj] = sim_scores_tgt_tmp
    if proj not in labels_tgt[config].keys():
        labels_tgt[config][proj] = labels_tgt_tmp

        # ROC json files
    roc_src_out_path = os.path.join(args.out_dir, "vexir2vec-roc-src-json")
    src_sim_scores_out = "vexir2vec-" + proj + "-sim-scores-" + config + ".json"
    src_labels_out = "vexir2vec-" + proj + "-labels-" + config + ".json"
    if not os.path.isdir(roc_src_out_path):
        os.makedirs(roc_src_out_path, exist_ok=True)
    print(src_sim_scores_out)
    with open(os.path.join(roc_src_out_path, src_sim_scores_out), "w") as dict_file:
        json.dump(sim_scores_src, dict_file)
    print(src_labels_out)
    with open(os.path.join(roc_src_out_path, src_labels_out), "w") as dict_file:
        json.dump(labels_src, dict_file)

    roc_tgt_out_path = os.path.join(args.out_dir, "vexir2vec-roc-tgt-json")
    tgt_sim_scores_out = "vexir2vec-" + proj + "-sim-scores-" + config + ".json"
    tgt_labels_out = "vexir2vec-" + proj + "-labels-" + config + ".json"
    if not os.path.isdir(roc_tgt_out_path):
        os.makedirs(roc_tgt_out_path, exist_ok=True)
    print(tgt_sim_scores_out)
    with open(os.path.join(roc_tgt_out_path, tgt_sim_scores_out), "w") as dict_file:
        json.dump(sim_scores_tgt, dict_file)
    print(tgt_labels_out)
    with open(os.path.join(roc_tgt_out_path, tgt_labels_out), "w") as dict_file:
        json.dump(labels_tgt, dict_file)

    return


if __name__ == "__main__":
    printVex2vec()
    parser = ArgumentParser(description="VexIR2Vec framework for binary similarity.")
    parser.add_argument(
        "-bmp", "--best_model_path", required=True, help="Path to the best model"
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        help="Directory containing data files of all the projects.",
    )
    group_gt = parser.add_mutually_exclusive_group(required=True)
    group_gt.add_argument(
        "-test_csv", "--test_csv_path", help="Path to ground truth csv"
    )
    group_gt.add_argument(
        "-search_gt_dir", "--binsearch_test_dir", help="Path to ground truth csv"
    )
    parser.add_argument(
        "-res_dir",
        "--res_dir",
        required=True,
        help="Path to output directory to store results (prec, recall, f1)",
    )
    parser.add_argument(
        "-out_dir",
        "--out_dir",
        required=True,
        help="Path to output directory of roc json files",
    )
    parser.add_argument(
        "-cfg",
        "--use_cfg",
        type=bool,
        default=False,
        help="Use CFG data for inferencing if set",
    )
    parser.add_argument(
        "-n",
        "--threads",
        type=bool,
        default=1,
        help="Perform binary search for ground truth functions",
    )
    parser.add_argument(
        "-chunks",
        "--num_chunks",
        type=int,
        default=100,
        help="Perform binary search for ground truth functions",
    )
    parser.add_argument(
        "--cosine_similarity", action="store_true", help="Use cosine similarity"
    )

    args = parser.parse_args()

    getTopkScores(args, args.test_csv_path)
