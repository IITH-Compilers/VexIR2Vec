# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Searching Script"""

# Usage/ flow for inference searching: v2v_search_wrapper.sh

# To use this script as standalone:
# Usage:  python /path/to/new_vexir2vec_searching.py -bmp /path/to/model -dp /path/to/data-files/ -search_gt_dir /path/to/searching-groundtruth/ -out_dir path/to/results/ -res_dir /path/to/results/ -n 20 -chunks 100


import os
import time
import torch
import sys
import random
import json
import numpy as np
import pandas as pd
import re

sys.path.append(os.path.abspath("../../embeddings/vexNet"))

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
from utils_inference import (
    kdtreeNeigh,
    getEmbedFromData,
    newGetKeyEmbedList,
    getKeyEmbedList,
    modifyKey,
)


import warnings

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

device = torch.device("cpu")
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance()
scaler = preprocessing.StandardScaler()

test_dict = {
    # "coreutils": [
    #     "who",
    #     "stat",
    #     "tee",
    #     "sha256sum",
    #     "sha384sum",
    #     "sha224sum",
    #     "base32",
    #     "sha512sum",
    #     "unexpand",
    #     "expand",
    #     "base64",
    #     "chroot",
    #     "md5sum",
    #     "env",
    #     "sha1sum",
    #     "uniq",
    #     "readlink",
    #     "fmt",
    #     "stty",
    #     "cksum",
    #     "head",
    #     "realpath",
    #     "uptime",
    #     "wc",
    #     "b2sum",
    #     "tr",
    #     "join",
    #     "numfmt",
    #     "factor",
    #     "split",
    #     "dd",
    #     "rm",
    #     "shred",
    #     "touch",
    # ],
    "diffutils": ["cmp", "sdiff"],
    # "findutils": ["xargs"],
    # "gzip": ["gzip"],
    # "lua": ["lua", "luac"],
    # "curl": ["curl"],
    # "putty": [
    #     "cgtest",
    #     "fuzzterm",
    #     "osxlaunch",
    #     "plink",
    #     "pscp",
    #     "psftp",
    #     "psocks",
    #     "psusan",
    #     "puttygen",
    #     "testcrypt",
    #     "testsc",
    #     "testzlib",
    #     "uppity",
    # ],
}


def generateSearchSpace(args):

    model = torch.load(args.best_model_path, map_location=device)
    model.eval()

    x86_data_path = args.data_path
    arm_data_path = x86_data_path.replace("x86", "arm")

    search_configs = [
        "x86-clang-12-O0",
        "x86-clang-12-O3",
        "x86-clang-8-O2",
        "x86-gcc-10-O2",
        "x86-gcc-8-O1",
        "x86-clang-12-O1",
        "x86-clang-8-O0",
        "x86-clang-8-O3",
        "x86-gcc-10-O0",
        "x86-gcc-10-O3",
        "x86-gcc-8-O2",
        "x86-clang-12-O2",
        "x86-clang-8-O1",
        "x86-gcc-10-O1",
        "x86-gcc-8-O0",
        "x86-gcc-8-O3",
    ]

    search_key_embed_dict = {}
    for search_config in search_configs:
        search_key_embed_dict[search_config] = []

    search_key_list = []
    search_key_allembed_list = []
    search_key_embed_list = []
    use_cfg = args.use_cfg

    search_start = time.time()
    for search_config in search_configs:
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
            print(search_config_data_path)
            for bin_name in test_dict[proj]:
                search_data_list = []
                stripped_data = (
                    os.path.join(search_config_data_path, bin_name) + ".data"
                )
                unstripped_data = (
                    os.path.join(
                        search_config_data_path.replace(STRIPPED_DIR, UNSTRIPPED_DIR),
                        bin_name,
                    )
                    + ".data"
                )
                # print(unstripped_data)
                if not os.path.exists(stripped_data) or not os.path.exists(
                    unstripped_data
                ):
                    continue
                with open(stripped_data, "r") as strip_file, open(
                    unstripped_data, "r"
                ) as unstrip_file:
                    search_data_list = getKeyEmbedList(
                        unstrip_file, strip_file, use_cfg
                    )
                    # print(search_data_list)
                    # exit(0)
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
                                    search_data[6],
                                    search_data[7],
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
                                    search_data[5],
                                    search_data[6],
                                )
                            )
        search_key_embed_dict[search_config] = search_config_key_embed_list

    for search_config in search_configs:
        search_key_allembed_list.extend(search_key_embed_dict[search_config])
    # print(search_key_allembed_list)
    key_list = list(map(lambda x: (x[0], x[1], x[2]), search_key_allembed_list))

    embed_list = list(map(lambda x: x[3], search_key_allembed_list))
    embedding = torch.stack(embed_list, dim=0)

    opcEmbed_list = list(map(lambda x: x[3], search_key_allembed_list))
    opcEmbed = torch.squeeze(torch.stack(opcEmbed_list, dim=0), 1)

    tyEmbed_list = list(map(lambda x: x[4], search_key_allembed_list))
    tyEmbed = torch.squeeze(torch.stack(tyEmbed_list, dim=0), 1)

    argEmbed_list = list(map(lambda x: x[5], search_key_allembed_list))
    argEmbed = torch.squeeze(torch.stack(argEmbed_list, dim=0), 1)

    strEmbed_list = list(map(lambda x: x[6], search_key_allembed_list))
    strEmbed = torch.squeeze(torch.stack(strEmbed_list, dim=0), 1)

    libEmbed_list = list(map(lambda x: x[7], search_key_allembed_list))
    libEmbed = torch.squeeze(torch.stack(libEmbed_list, dim=0), 1)

    if use_cfg:

        cfgEmbed_list = list(
            map(
                lambda x: torch.tensor(
                    padAndSlice(x[6], CNN_INP_DIM), dtype=torch.float
                ),
                search_key_allembed_list,
            )
        )
        cfgEmbed = torch.stack(cfgEmbed_list, dim=0)
        dataset = TensorDataset(embedding, strEmbed, libEmbed, cfgEmbed)
        train_dataloader = DataLoader(
            dataset, batch_size=2048, shuffle=False, num_workers=20
        )
        for train_data in train_dataloader:
            with torch.no_grad():
                output = model(
                    train_data[0], train_data[1], train_data[2], train_data[3]
                )
            search_key_embed_list.extend(output.tolist())
        # print(search_key_embed_list)
    else:
        dataset = TensorDataset(opcEmbed, tyEmbed, argEmbed, strEmbed, libEmbed)
        train_dataloader = DataLoader(
            dataset, batch_size=2048, shuffle=False, num_workers=20
        )
        for train_data in train_dataloader:
            with torch.no_grad():
                output, _ = model(
                    train_data[0],
                    train_data[1],
                    train_data[2],
                    train_data[3],
                    train_data[4],
                    test=True,
                )
            search_key_embed_list.extend(
                [np.array(embed).astype(np.float32) for embed in output.tolist()]
            )

    search_key_embed_list = list(zip(key_list, search_key_embed_list))
    # print(search_key_embed_list)
    search_key_embed_dict = {}

    for tup in search_key_embed_list:
        if tup[0][0] not in search_key_embed_dict.keys():
            # Config
            search_key_embed_dict[tup[0][0]] = {}
            # Config, addr -> (Key, embed)
        search_key_embed_dict[tup[0][0]][tup[0][1]] = (tup[0][2], tup[1])

    search_end = time.time()
    print("Search time: ", search_end - search_start)

    return search_key_embed_dict

    # Get Search Scores


def getSearchScores(
    args, test_dir, num_chunks, search_key_embed_dict, package, configs
):
    print(f"Package Name: {package }")

    project = "vexir2vec"
    project_dir = os.path.join(args.res_dir, project)
    package_res_dir = os.path.join(project_dir, package)
    if not os.path.exists(package_res_dir):
        os.makedirs(package_res_dir, exist_ok=True)

    gt_configs = configs

    optimization_levels = ["O0", "O1", "O2", "O3"]

    results_dict = {
        config: {opt_level: None for opt_level in optimization_levels}
        for config in gt_configs
    }

    for gt_config in gt_configs:
        gt_config_path = os.path.join(test_dir, gt_config)
        matches = 0
        total_gt = 0

        for csv_test in os.listdir(gt_config_path):

            csv_test = os.path.join(gt_config_path, csv_test)
            config = os.path.basename(csv_test).split(".")[0].split("-", 1)[1]

            tp_data = pd.read_csv(csv_test)

            missing_query_points = 0

            ap_list = []

            query_key_embed_list = []
            query_key_embed_list_full = []
            search_key_embed_list = []
            base_config = config

            res_file = os.path.join(package_res_dir, f"{base_config }.results")
            if os.path.isfile(res_file):
                print(f"{res_file } exists.")
                return
            query_start = time.time()
            for key in search_key_embed_dict.keys():

                if config not in key:
                    for addr_key in search_key_embed_dict[key].keys():
                        search_key_embed_list.append(
                            search_key_embed_dict[key][addr_key]
                        )

            search = np.array([tup[1] for tup in search_key_embed_list]).astype(
                np.float32
            )

            search_key_list = [i[0] for i in search_key_embed_list]
            search_key_list = np.array(search_key_list)

            projects_to_filter = []

            projects_to_filter.append(package)

            tp_data = tp_data[tp_data["proj"].isin(projects_to_filter)]

            for _, test_row in tp_data.iterrows():
                proj = test_row["proj"]
                bin_name = test_row["bin_name_" + config]
                addr = test_row["address_" + config]
                # exit(0)
                if (
                    config + proj + bin_name in search_key_embed_dict
                    and addr in search_key_embed_dict[config + proj + bin_name]
                ):
                    query_key_embed_list_full.append(
                        search_key_embed_dict[config + proj + bin_name][addr]
                    )
                elif proj in test_dict.keys():
                    missing_query_points += 1
                else:
                    with open("missing_keys.txt", "a") as file:
                        print(
                            "Keys not found:", config + proj + bin_name, addr, file=file
                        )

            query_end = time.time()
            print("Query time: ", query_end - query_start)
            kdt = kdtreeNeigh(search)
            start_chunk_index = 0
            end_chunk_index = 0
            chunk_scores_start = time.time()
            while True:
                start_chunk_index = end_chunk_index
                end_chunk_index += int(tp_data.shape[0] / num_chunks)
                # print(end_chunk_index,len(query_key_embed_list_full))
                # exit(0)
                if start_chunk_index >= len(query_key_embed_list_full):
                    break
                query_key_embed_list = query_key_embed_list_full[
                    start_chunk_index:end_chunk_index
                ]
                query = np.array([tup[1] for tup in query_key_embed_list]).astype(
                    np.float32
                )

                query_key_list = [i[0] for i in query_key_embed_list]

                vexir2vec_func_index, vexir2vec_func_dist = kdt.getTopkNeighAndDist(
                    query, NUM_NEIGHBORS
                )
                # print(vexir2vec_func_index)
                # print(vexir2vec_func_dist)
                # exit(0)
                reciprocal_rank_sum = 0

                calc_scores_start = time.time()
                vexir2vec_func_index = np.array(vexir2vec_func_index)
                vexir2vec_func_dist = np.array(vexir2vec_func_dist)

                with open(res_file, "a") as f:

                    for index in range(len(vexir2vec_func_index)):
                        matches_new = 0
                        avg_prec = 0
                        prev_dist = -1
                        rank = 0
                        last_rank = rank
                        first_match = 0
                        arr = vexir2vec_func_index[index]
                        keymatch_arr = search_key_list[arr] == query_key_list[index]
                        for label_index, label in enumerate(keymatch_arr):
                            rank += 1
                            if label and last_rank != rank:
                                matches_new += 1
                                avg_prec += matches_new / rank
                                last_rank = rank
                                if first_match == 0:
                                    reciprocal_rank_sum += 1 / rank
                                    matches += 1
                                    first_match = 1
                        if matches_new:
                            avg_prec /= matches_new
                        total_gt += 1
                        ap_list.append(avg_prec)
                calc_scores_end = time.time()
                print("Calc scores time: ", calc_scores_end - calc_scores_start)

            chunk_scores_end = time.time()
            print("Chunk scores time: ", chunk_scores_end - chunk_scores_start)

            mean_ap = sum(ap_list) / len(ap_list)
            acc = (matches / total_gt) * 100
            mrr = reciprocal_rank_sum / total_gt

            with open(res_file, "a") as f:
                f.write("Config: " + base_config + "\n")
                f.write("MAP: " + str(mean_ap) + "\n")
                f.write("Acc: " + str(acc) + "\n")
                f.write("MRR: " + str(mrr) + "\n")
            print("MAP: ", mean_ap)
            print("Acc: ", acc)
            print("MRR: ", mrr)

            print(f"Missing query points : {missing_query_points }")

            # Store the MAP value in the results dictionary
            config_name = gt_config.replace("-", "")
            opt_level = base_config.split("-")[-1]
            results_dict[gt_config][opt_level] = round(mean_ap, 2)

    return results_dict


if __name__ == "__main__":
    printVex2vec()
