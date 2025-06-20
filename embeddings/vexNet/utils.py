# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Main helper script to define paths and the helper functions"""

import os
import glob
import time
import h5py
import torch
import random
import pickle, json
import numpy as np
import pandas as pd
import ast
import json
import fasttext
import fasttext.util
import math
from scipy.stats import rankdata
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from pyfiglet import figlet_format
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
import angr
import string

NUM_SB = 1
SEED = 1004
PATIENCE = 5
INP_DIM = 128

OUT_DIM = 100
CNN_INP_DIM = 150

NUM_WALKS = 1
NUM_NEIGHBORS = 10

H5_PATH = "/path/to/h5-directory/"

KEYS_FILE = os.path.join(
    H5_PATH,
    "/path/to/keys.h5",
)

TRAIN_DATA_FILE = os.path.join(H5_PATH, "/path/to/training-data.h5")
STRIPPED_DIR = "/path/to/stripped-directory"
UNSTRIPPED_DIR = "/path/to/unstripped-directory"


DB_PATH = "/path/to/db"


DATA_PATH = "/path/to/outermost-directory-containing-data-files/"
X86_DATA_PATH = os.path.join(DATA_PATH, "x86-data-all")
ARM_DATA_PATH = os.path.join(DATA_PATH, "arm-data-all")
FASTTEXT_PATH = "../cc.en.300.bin"

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
ft = None
np.set_printoptions(linewidth=100000)
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance()
scaler = preprocessing.StandardScaler()


def loadFasttextInit():
    global ft
    ft = fasttext.load_model(FASTTEXT_PATH)
    fasttext.util.reduce_model(ft, 100)


def printVex2vec():
    print(figlet_format("VexIR2Vec", font="starwars", width=200))


def genDataForVisualization(args):
    device = args.device
    model = torch.load(args.best_model_path).to(device)
    data = loadCSVAndGendataforotm(args.test_path)
    data_list = list(data.itertuples(index=False, name=None))
    labels, embeds = [], []
    model.eval()
    with torch.no_grad():
        for tup in data_list:
            labels.append(tup[0])
            embed = torch.from_numpy(tup[1]).float().to(device).view(-1, 1, NUM_SB, 100)
            embeds.append(torch.flatten(model(embed)).cpu().numpy())
    emb = np.array(embeds)
    np.savetxt("embed.tsv", emb, delimiter="\t", fmt="%f")
    lab = list(le.fit_transform(labels))
    print(lab)
    np.savetxt("labels.csv", lab, delimiter="\t", fmt="%f")

    print("Files generated.")
    return


def getSbNpArray(series):
    return series.apply(
        lambda x: np.vstack(
            [
                np.fromstring(
                    i.replace("[", "").replace("]", "").replace(" ", "  "), sep=" "
                )
                for i in eval(x)
            ]
        )
    )


def strToMuldimNpArr(s):

    try:
        arr = np.array(json.loads(s), dtype=float)
    except Exception as e:
        print(e)
        arr = np.zeros((1, 1))

    return arr


def rwStrToArr(s):
    # Remove unnecessary characters from the string
    cleaned_string = s.replace("[", "").replace("]", "").replace("'", "")

    # Split the string into individual elements
    elements = cleaned_string.split(", ")

    # Convert elements to floats
    float_array = np.array([list(map(float, row.split())) for row in elements])

    return float_array


def getEmbedding(strRefs):  # generates vectors for inference
    vectors = [ft.get_word_vector(word).reshape((1, 100)) for word in strRefs]
    return np.sum(np.concatenate(vectors, axis=0), axis=0)


def getExtlibemb(extlib):
    extlibs = []
    if type(extlib) == str:
        extlibs = extlib.split("^")
    libVec = getEmbedding(extlibs) if extlibs else np.zeros(100)
    return libVec.reshape((1, -1))


def getStremb(strRef):  # generates vectors for inference
    strRefs = []
    if type(strRef) == str:
        strRefs = strRef.split("^")
    strVec = getEmbedding(strRefs) if strRefs else np.zeros(100)
    return strVec


def loadCSVAndGendataforotm(csv_filepath):
    d = pd.read_csv(csv_filepath)
    d.drop(["Unnamed: 0"], axis=1, inplace=True)
    print("File loaded.")
    print(d.shape)
    # Uncomment this if block for generating data for visualization
    random.seed(100)

    if len(d.key_unst.unique()) > 10:
        rndm_keys = random.sample(list(d.key_unst.unique()), 10)
        print(rndm_keys)

        d = d[d.key_unst.isin(rndm_keys)]

    print(d.shape)
    print(len(d.key_unst.unique()))
    src_cols = [col for col in d.columns if col.startswith("src_embed_")]
    tgt_cols = [col for col in d.columns if col.startswith("tgt_embed_")]

    d["embed_st_src"] = d.loc[:, src_cols].values.tolist()
    d.embed_st_src = d.embed_st_src.apply(lambda x: np.array(x).reshape((NUM_SB, 100)))

    d["embed_st_tgt"] = d.loc[:, tgt_cols].values.tolist()
    d.embed_st_tgt = d.embed_st_tgt.apply(lambda x: np.array(x).reshape((NUM_SB, 100)))

    d.drop(src_cols + tgt_cols, axis=1, inplace=True)

    tmp = d.iloc[:, [0, 1]].rename(columns={"embed_st_src": "embed_st"})
    tmp2 = d.iloc[:, [0, 2]].rename(columns={"embed_st_tgt": "embed_st"})

    data = pd.concat([tmp, tmp2], axis=0, ignore_index=False)

    data.key_unst = le.fit_transform(data.key_unst.values)
    data.reset_index(drop=True, inplace=True)
    print(data.key_unst.value_counts())
    data = data[~data.key_unst.isin([2, 9])]
    return data


def modifyKey(keys):

    modified_keys = []
    for idx, key in enumerate(keys):
        x = key.decode()
        if x == "nan":
            print(idx, x)
            continue
        x = re.split(", |\)", x)
        # if len(x)>2:
        modified_str = x[0] + ")" + x[2]
        bytes(modified_str, encoding="utf-8")
        modified_keys.append(bytes(modified_str, encoding="utf-8"))
    return np.array(modified_keys)


def pregetStrembH5Dataset(file, use_cfg=False):
    print("file: ", file)
    hf = h5py.File(file, "r")
    keys = hf.get("keys")[()]

    keys = modifyKey(keys)
    print("keys.shape: ", keys.shape)
    opc_embeds = hf.get("opc_embed")[()]
    type_embeds = hf.get("type_embed")[()]
    arg_embeds = hf.get("arg_embed")[()]

    print("opc_embeds.shape: ", opc_embeds.shape)
    print("type_embeds.shape: ", type_embeds.shape)
    print("arg_embeds.shape: ", arg_embeds.shape)

    strEmbeds = hf.get("strRefs")[()]
    print("strEmbeds.shape: ", strEmbeds.shape)

    libEmbeds = hf.get("extlibs")[()]
    print("libEmbeds.shape: ", libEmbeds.shape)

    if not use_cfg:
        return keys, opc_embeds, type_embeds, arg_embeds, strEmbeds, libEmbeds

    cfgs = hf.get("cfgs")[()]
    print("cfgs.shape: ", cfgs.shape)
    print(type(cfgs[0]))
    print(type(cfgs[1]))
    print(type(cfgs[2]))

    print(f"{file } loaded.")

    return keys, opc_embeds, type_embeds, arg_embeds, strEmbeds, libEmbeds, cfgs


def pregetStrembH5Cfg(cfg_dir):
    extension = ".h5"
    print(os.listdir(cfg_dir))
    merged_df = pd.DataFrame(columns=["cfgs"])
    merged_cfgs = merged_df.cfgs  # Series
    for filename in os.listdir(cfg_dir):
        if filename.endswith(extension):
            print("getStrembing - ", filename)
            file = os.path.join(cfg_dir, filename)
            try:
                cfg = pd.read_hdf(file, "cfgs")
                merged_cfgs = pd.concat([merged_cfgs, cfg], ignore_index=True)  # Series
            except IndexError as IE:
                print("Error: ", IE)

    merged_cfgs = np.array(merged_cfgs.values.tolist())
    print("merged_cfgs shape: ", merged_cfgs.shape)
    return merged_cfgs


def padAndReshapeArray(
    x, sz, INP_DIM
):  # func getting used earlier for fixing no. of SBs per func
    shape = x.shape[0]
    if shape <= sz:
        reqd = sz - shape
        padded_array = np.zeros((reqd, INP_DIM))
        tmp_arr = np.concatenate((x, padded_array), axis=0)

    else:
        if shape > sz:
            tmp_arr = x[:sz, :]

    return tmp_arr

    # this fn is for padding /slicing iput matrix for CNN
    # considers input to be a square matrix


def padAndSlice(x, CNN_INP_DIM):
    # Pad the input matrix with 0s if its size is less than the threshold size
    if x.shape[0] < CNN_INP_DIM:
        pad_size = CNN_INP_DIM - x.shape[0]
        x = np.pad(x, ((0, pad_size), (0, pad_size)), mode="constant")

        # Reshape the input matrix if its size is greater than the threshold size
    if x.shape[0] > CNN_INP_DIM:
        new_shape = (CNN_INP_DIM, CNN_INP_DIM)
        x = x[: new_shape[0], : new_shape[1]]

        # return the padded or reshaped input matrix
    return x


def mergeEmb(x):
    new_emb = x.sum(axis=0)
    return new_emb

    # Use label as 0 for similar pairs in contrastive loss and 1 in case of binary cross-entropy loss


def genSiamesePairs(
    embed_x, embed_y, stremb_x, stremb_y, extlib_x, extlib_y, pairs_list, similar
):
    # appending positive pairs to pos_list
    if similar:
        pairs_list.append(
            (
                torch.from_numpy(embed_x),
                torch.from_numpy(stremb_x),
                torch.from_numpy(extlib_x),
                torch.from_numpy(embed_y),
                torch.from_numpy(stremb_y),
                torch.from_numpy(extlib_y),
                torch.from_numpy(np.array(1)),
            )
        )

        # appending negative pairs to neg_list
    else:
        pairs_list.append(
            (
                torch.from_numpy(embed_x),
                torch.from_numpy(stremb_x),
                torch.from_numpy(extlib_x),
                torch.from_numpy(embed_y),
                torch.from_numpy(stremb_y),
                torch.from_numpy(extlib_y),
                torch.from_numpy(np.array(0)),
            )
        )


def genTrainDataPairsNew(
    train_list,
):  # train_list=h5 file now (that contains similar pairs)
    pos_pairs, neg_pairs = [], []

    column_names = [
        "src_embedding",
        "tgt_embedding",
        "src_strRefs",
        "tgt_strRefs",
        "src_extlibs",
        "tgt_extlibs",
        "src_key",
        "tgt_key",
    ]
    # column_names = ['src_embedding', 'tgt_embedding', 'src_key', 'tgt_key', ]
    df = pd.DataFrame(columns=column_names)
    h5f = h5py.File(train_list, "r")
    c1 = h5f.get("src_keys")
    c2 = h5f.get("src_strRefs")
    c3 = h5f.get("src_extlibs")
    c4 = h5f.get("src_embeddings")
    c5 = h5f.get("tgt_keys")
    c6 = h5f.get("tgt_strRefs")
    c7 = h5f.get("tgt_extlibs")
    c8 = h5f.get("tgt_embeddings")
    df["src_key"] = list(c1[:])
    df["src_strRefs"] = list(c2[:])
    df["src_extlibs"] = list(c3[:])
    df["src_embedding"] = list(c4[:])
    df["tgt_key"] = list(c5[:])
    df["tgt_strRefs"] = list(c6[:])
    df["tgt_extlibs"] = list(c7[:])
    df["tgt_embedding"] = list(c8[:])

    rndm_idxs = random.sample(range(len(df)), len(df))

    for idx in range(len(df)):
        row = df[idx : idx + 1]
        genSiamesePairs(
            row.iloc[0][0],
            row.iloc[0][1],
            row.iloc[0][2],
            row.iloc[0][3],
            row.iloc[0][4],
            row.iloc[0][5],
            pos_pairs,
            similar=1,
        )

        genSiamesePairs(
            row.iloc[0][1],
            row.iloc[0][0],
            row.iloc[0][3],
            row.iloc[0][2],
            row.iloc[0][5],
            row.iloc[0][4],
            pos_pairs,
            similar=1,
        )

        for i in range(1):
            if rndm_idxs[idx] == idx and idx < len(df):
                rndm_idxs[idx], rndm_idxs[idx + 1] = rndm_idxs[idx + 1], rndm_idxs[idx]
            if not idx + i < len(df):
                continue
            dissim_idx = rndm_idxs[idx + i]
            dissim_row = df[dissim_idx : dissim_idx + 1]

            if row.iloc[0][6] != dissim_row.iloc[0][6]:
                genSiamesePairs(
                    row.iloc[0][0],
                    dissim_row.iloc[0][0],
                    row.iloc[0][2],
                    dissim_row.iloc[0][2],
                    row.iloc[0][4],
                    dissim_row.iloc[0][4],
                    neg_pairs,
                    similar=0,
                )
                genSiamesePairs(
                    row.iloc[0][0],
                    dissim_row.iloc[0][1],
                    row.iloc[0][2],
                    dissim_row.iloc[0][3],
                    row.iloc[0][4],
                    dissim_row.iloc[0][5],
                    neg_pairs,
                    similar=0,
                )

    train_data = pos_pairs + neg_pairs
    return train_data, pos_pairs, neg_pairs


def collectTestBins():
    test_dict = {}
    test_dict["findutils"] = ["xargs.data"]

    test_dict["diffutils"] = ["cmp.data", "sdiff.data"]

    test_dict["coreutils"] = [
        "who.data",
        "stat.data",
        "tee.data",
        "sha256sum.data",
        "sha384sum.data",
        "sha224sum.data",
        "base32.data",
        "sha512sum.data",
        "unexpand.data",
        "expand.data",
        "base64.data",
        "chroot.data",
        "env.data",
        "sha1sum.data",
        "uniq.data",
        "readlink.data",
        "fmt.data",
        "stty.data",
        "cksum.data",
        "head.data",
        "realpath.data",
        "uptime.data",
        "wc.data",
        "b2sum.data",
        "tr.data",
        "join.data",
        "numfmt.data",
        "factor.data",
        "split.data",
        "dd.data",
        "rm.data",
        "shred.data",
        "touch.data",
    ]

    return test_dict


def loadFiles(st_path, test_dict, use_cfg):

    file_dfs = []

    for st_file in sorted(os.listdir(st_path)):

        proj = os.path.basename(os.path.dirname(os.path.dirname(st_path)))

        if st_file.endswith(".csv"):
            if (
                st_file.replace(".csv", ".data") not in test_dict[proj]
                and "test" not in st_file
            ):

                if use_cfg == True:
                    col_names_str = [
                        "addr",
                        "key",
                        "fnName",
                        "strRefs",
                        "extlibs",
                        "embed",
                        "cfg",
                    ]
                else:
                    col_names_str = [
                        "addr",
                        "key",
                        "strRefs",
                        "extlibs",
                        "embed_O",
                        "embed_T",
                        "embed_A",
                    ]

                st_df = pd.read_csv(
                    os.path.join(st_path, st_file),
                    sep="\t",
                    skiprows=[0],
                    names=col_names_str,
                    header=None,
                )

                st_df.dropna(axis=0, inplace=True)
                st_df.reset_index(drop=True, inplace=True)

                file_dfs.append(st_df)

    return file_dfs

    # def load_datasets(projects, test_dict, srcData):


def loadDatasets(proj_path, test_dict, use_cfg, hf=None):
    st_path_list = []

    if os.path.isdir(proj_path):

        for config in os.listdir(proj_path):
            data_path = os.path.join(proj_path, config)
            st_path_list.append(os.path.join(data_path, STRIPPED_DIR))

    dfs = []
    for st_path in st_path_list:
        try:
            dataset_df = pd.concat(loadFiles(st_path, test_dict, use_cfg))
            if hf != None:
                hf.write_data(dataset_df)
            else:
                dfs.append(dataset_df)
        except Exception as err:
            print("Error in loading data for {}".format(st_path))
            print(err)

    return dfs


def loadData(proj_path, test_dict, use_cfg, hf=None):
    if use_cfg == True:
        print("Using data files merged with cfgs")
    if hf != None:
        return loadDatasets(proj_path, test_dict, use_cfg, hf)
    return pd.concat(loadDatasets(proj_path, test_dict, use_cfg))


def checkAndCalcprfscore(tp, fp, fn):
    try:
        precision = round(tp / (tp + fp), 4)
    except ZeroDivisionError:
        precision = "NaN"

    try:
        recall = round(tp / (tp + fn), 4)
    except ZeroDivisionError:
        recall = "NaN"

    try:
        f1 = round(((2 * precision * recall) / (precision + recall)), 4)
    except ZeroDivisionError:
        f1 = "NaN"

    return precision, recall, f1


def forwardPass(device, model, test_pairs, best_model_path, config_name):
    matches, mismatches = 0, 0
    model.eval()
    with torch.no_grad():
        for row in test_pairs:

            fnEmbed_x, fnEmbed_y = torch.from_numpy(row[0]), torch.from_numpy(row[1])
            fnEmbed_x, fnEmbed_y = fnEmbed_x.view(
                -1, 1, NUM_SB, INP_DIM
            ), fnEmbed_y.view(-1, 1, NUM_SB, INP_DIM)

            output1 = model(fnEmbed_x.float().to(device))
            output2 = model(fnEmbed_y.float().to(device))
            with open(
                os.path.dirname(best_model_path)
                + "/{}-L1.classifier".format(config_name),
                "rb",
            ) as cls:
                classifier = pickle.load(cls)

            pred = classifier.predict(
                torch.abs(torch.sub(output1, output2)).cpu().numpy().reshape((1, -1))
            )

            if pred == 1:
                matches += 1
            else:
                mismatches += 1

    return matches, mismatches


def collectNumbersPerConfig(
    device, model, configs, config, best_model_path, config_name
):
    vals = configs[config]
    # EXP-1
    optX, optY = tuple(config.rsplit("-", 2)[-2:])

    if "incorrect" in os.path.basename(vals[1]):
        corr, incorr = vals[0], vals[1]
    else:
        corr, incorr = vals[1], vals[0]

    cols = ["emb_" + optX, "emb_" + optY]
    print(cols)

    corr_df, incorr_df = pd.read_csv(corr, usecols=cols), pd.read_csv(
        incorr, usecols=cols
    )
    corr_df[cols[0]] = getSbNpArray(corr_df[cols[0]])
    corr_df[cols[0]] = corr_df[cols[0]].apply(lambda x: padAndReshapeArray(x, NUM_SB))

    corr_df[cols[1]] = getSbNpArray(corr_df[cols[1]])
    corr_df[cols[1]] = corr_df[cols[1]].apply(lambda x: padAndReshapeArray(x, NUM_SB))

    incorr_df[cols[0]] = getSbNpArray(incorr_df[cols[0]])
    incorr_df[cols[0]] = incorr_df[cols[0]].apply(
        lambda x: padAndReshapeArray(x, NUM_SB)
    )

    incorr_df[cols[1]] = getSbNpArray(incorr_df[cols[1]])
    incorr_df[cols[1]] = incorr_df[cols[1]].apply(
        lambda x: padAndReshapeArray(x, NUM_SB)
    )

    corr_list = list(corr_df.itertuples(index=False, name=None))
    incorr_list = list(incorr_df.itertuples(index=False, name=None))

    tp, fn = forwardPass(device, model, corr_list, best_model_path, config_name)
    fp, tn = forwardPass(device, model, incorr_list, best_model_path, config_name)
    precision, recall, f1 = checkAndCalcprfscore(tp, fp, fn)

    resultStr = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
        config, tp, fp, tn, fn, precision, recall, f1
    )

    return resultStr


def getTopkNeighbors(tp_src, tp_tgt, pri_addr_list, sec_addr_list, temperature):
    vexir2vec_func_list = []
    distances = torch.cdist(torch.from_numpy(tp_src), torch.from_numpy(tp_tgt))
    rd = rankdata(distances, axis=1, method="dense")
    mask = (rd <= NUM_NEIGHBORS) & (distances.cpu().numpy() <= temperature)
    # print(mask)
    ridx, cidx = np.where(mask == True)
    # print(np.unique(ridx, return_counts=True)[1])
    dic = {}
    for i in range(len(ridx)):
        if ridx[i] not in dic:
            dic[ridx[i]] = [(pri_addr_list[ridx[i]], sec_addr_list[cidx[i]])]
        else:
            dic[ridx[i]].append((pri_addr_list[ridx[i]], sec_addr_list[cidx[i]]))

    _, i = np.unique(ridx, return_index=True)
    sec_addr_arr = np.array(sec_addr_list)
    sec_addr_lookup = sec_addr_arr[cidx]
    sec_addr_lookup = np.split(sec_addr_lookup, i[1:], axis=0)

    idx = 0
    for row in sec_addr_lookup:
        dummy_x = np.full(row.shape, pri_addr_list[idx])
        idx += 1
        vexir2vec_func_list.append(zip(dummy_x, row))

    return vexir2vec_func_list


def savePlot(iteration, loss, filename):
    plt.plot(iteration, loss)
    plt.savefig("./lossplots/" + filename)


def removeEntity(text, entity_list):
    for entity in entity_list:
        if entity in text:
            text = text.replace(entity, " ")
    return text.strip()


def functionTraversal(
    project,
    nodes,
    isUnstripped,
    func_norm,
    skipFuncs,
    funcs,
    addr_to_line=None,
):
    for fn_addr in nodes:
        fnName = project.kb.functions.function(fn_addr).name
        isUserFunc = False

        if isUnstripped:
            if addr_to_line is not None:
                isUserFunc = (
                    fn_addr in addr_to_line
                    and fnName not in skipFuncs
                    and not fnName.startswith("sub_")
                )
            else:
                isUserFunc = fnName not in skipFuncs and not fnName.startswith("sub_")
        else:
            isUserFunc = fnName.startswith("sub_") or fnName == "main"

        if isUserFunc:
            if project.kb.functions.function(fn_addr) is None:
                continue
            else:
                if not func_norm.isNotUsefulFunction(project.kb.functions[fnName]):
                    funcs.append(project.kb.functions[fnName])

    return sorted(funcs, key=lambda func: func.addr)


def processStringReferences(func, isUnstripped):
    strRefs = []
    try:
        for _, strRef in func.string_references():
            if isUnstripped:
                break
            """
            preprocess stringrefs for cleaning
            1. removing everything other than alphabets
            2. removing strings containing paths
            3. removing format specifiers
            4. lowercasing everything
            5. convert to separate tokens
            """
            # print("Debug StrRef: ",strRef)
            strRef = strRef.decode("latin-1")
            if "http" in strRef:
                continue
            format_specifiers = [
                "%c",
                "%s",
                "%hi",
                "%h",
                "%Lf",
                "%n",
                "%d",
                "%i",
                "%o",
                "%x",
                "%p",
                "%f",
                "%u",
                "%e",
                "%E",
                "%%",
                "%#lx",
                "%lu",
                "%ld",
                "__",
                "_",
            ]
            punctuations = list(string.punctuation)
            strRef = removeEntity(strRef, format_specifiers)
            strRef = removeEntity(strRef, punctuations)
            strRef = re.sub("[^a-zA-Z]", " ", strRef).lower().strip().split()
            if strRef:
                strRefs.extend(strRef)
    except angr.errors.SimEngineError as e:
        print("CONTINUING IN EXCEPT")
        pass
        # continue
    return "^".join(strRefs)


def processExtlibCalls(
    func_addr_list,
    funcs,
    project,
    ext_lib_functions,
    ext_lib_fn_names,
    extern_edges,
    edges,
    isCalled,
):
    for func in funcs:
        call_sites = func.get_call_sites()
        callee_func_list = [
            project.kb.functions.function(func.get_call_target(call_site))
            for call_site in call_sites
        ]
        extern_addr_list = []
        callee_addr_list = []
        for callee in callee_func_list:
            if callee is None:
                continue
            if callee.name in ext_lib_functions:
                if func.addr in ext_lib_fn_names:
                    ext_lib_fn_names[func.addr].append(callee.name)
                else:
                    ext_lib_fn_names[func.addr] = [callee.name]
            if callee.addr not in func_addr_list:
                extern_addr_list.append(callee.addr)
            elif callee.addr != func.addr:
                callee_addr_list.append(callee.addr)
        extern_edges[func.addr] = extern_addr_list
        edges[func.addr] = callee_addr_list
        if callee_addr_list:
            isCalled.update(callee_addr_list)


def preprocessH5Dataset(file, use_cfg=False):
    print("file: ", file)
    hf = h5py.File(file, "r")
    keys = hf.get("keys")[()]
    # print(keys)
    # exit(0)
    keys = modifyKey(keys)
    print("keys.shape: ", keys.shape)
    opc_embeds = hf.get("opc_embed")[()]
    type_embeds = hf.get("type_embed")[()]
    arg_embeds = hf.get("arg_embed")[()]

    print("opc_embeds.shape: ", opc_embeds.shape)
    print("type_embeds.shape: ", type_embeds.shape)
    print("arg_embeds.shape: ", arg_embeds.shape)

    strEmbeds = hf.get("strRefs")[()]
    # strEmbeds = preprocess_data(strEmbeds, 'str')
    print("strEmbeds.shape: ", strEmbeds.shape)

    libEmbeds = hf.get("extlibs")[()]
    # libEmbeds = preprocess_data(libEmbeds, 'lib')
    print("libEmbeds.shape: ", libEmbeds.shape)

    if not use_cfg:
        return keys, opc_embeds, type_embeds, arg_embeds, strEmbeds, libEmbeds

    cfgs = hf.get("cfgs")[()]
    print("cfgs.shape: ", cfgs.shape)
    print(type(cfgs[0]))
    print(type(cfgs[1]))
    print(type(cfgs[2]))

    print(f"{file} loaded.")

    return keys, opc_embeds, type_embeds, arg_embeds, strEmbeds, libEmbeds, cfgs


def preprocessH5Cfg(cfg_dir):
    extension = ".h5"
    print(os.listdir(cfg_dir))
    merged_df = pd.DataFrame(columns=["cfgs"])
    merged_cfgs = merged_df.cfgs  # Series
    for filename in os.listdir(cfg_dir):
        if filename.endswith(extension):
            print("processing - ", filename)
            file = os.path.join(cfg_dir, filename)
            try:
                cfg = pd.read_hdf(file, "cfgs")
                merged_cfgs = pd.concat([merged_cfgs, cfg], ignore_index=True)  # Series
            except IndexError as IE:
                print("Error: ", IE)

    merged_cfgs = np.array(merged_cfgs.values.tolist())
    print("merged_cfgs shape: ", merged_cfgs.shape)
    return merged_cfgs
