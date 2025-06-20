# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Main driver script for embedding generation and triplets generation"""

# Usage: python driver.py -b /path/to/binary -o /path/to/output.data -v /path/to/seed/embedding -d -n <Normalisation> -fchunks <no-of-func-chunks> -edb /path/to/DB_path/during/db/genaration/.db -mode <db/non-db>
import json
from colorama import Fore, Style
import pickle
from posixpath import basename
import os
import sys

SCRIPT_DIR_1 = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR = os.path.join(SCRIPT_DIR_1, "./embedding-gen")
sys.path.append(os.path.normpath(IMPORT_DIR))
import triplets as T
import argparse
import angr
import pdb
import re
import glob
import pyvex as py
import numpy as np
import pickle
from angr.analyses.forward_analysis.visitors.call_graph import CallGraphVisitor as CGV
from multiprocessing import Manager, Lock, Queue, Process
from pebble import ProcessPool
from functools import partial
import string
import time, sys, psutil
from collections import defaultdict
from function_normalizer import functionNormalizer
from angr.angrdb.db import AngrDB
import networkx as nx
import scipy.sparse
import pathlib
import sys
import logging
import sqlalchemy as db
from sqlalchemy.orm import scoped_session, sessionmaker

SCRIPT_DIR_2 = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR_1 = os.path.join(SCRIPT_DIR_2, "./db-gen")
sys.path.append(os.path.normpath(IMPORT_DIR_1))
from binary_db_model import binaryMetaData, binaryMetaDataUnstripped
import pickle
import pandas as pd
import embeddings as emb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VEXNET_PATH = os.path.join(SCRIPT_DIR, "../vexNet")
sys.path.append(os.path.normpath(VEXNET_PATH))
# print(sys.path)
import warnings
from utils import *

warnings.filterwarnings("ignore")

skipFuncs = [
    "_init",
    "UnresolvableCallTarget",
    "_start",
    "_dl_relocate_static_pie",
    "deregister_tm_clones",
    "UnresolvableJumpTarget",
    "register_tm_clones",
    "frame_dummy",
    "_fini",
]
skipLibs = ["libstdc++.so"]
freq_dict = {}
freq_weight_dict = None

# Caller -> [Callee1, Callee2, ...] map
edges = {}
# For external functions
extern_edges = {}
# Function -> [prefix, strRefs, sbVecs] map
func_vec_map = {}
# Set of functions that are called
isCalled = set()
ext_lib_fn_names = {}
str_fn_embs = {}
cgv_nodes = None
unstr_df = None


def removeEntity(text, entity_list):
    for entity in entity_list:
        if entity in text:
            text = text.replace(entity, " ")
    return text.strip()


def addCalleeVecs(orig_addr, edges, func_vec_map, ext_lib_fn_names):
    map_entry = func_vec_map[orig_addr]
    for callee_addr in edges:
        map_entry[1] = map_entry[1] + func_vec_map[callee_addr][1]
        if orig_addr in ext_lib_fn_names and callee_addr in ext_lib_fn_names:
            ext_lib_fn_names[orig_addr] = (
                ext_lib_fn_names[orig_addr] + ext_lib_fn_names[callee_addr]
            )
        elif (
            callee_addr in ext_lib_fn_names
        ):  # check needed as there can be no external lib func for a func
            ext_lib_fn_names[orig_addr] = ext_lib_fn_names[callee_addr]

        map_entry[2] += func_vec_map[callee_addr][2]  # for func level O embed
        map_entry[3] += func_vec_map[callee_addr][3]  # for func level T embed
        map_entry[4] += func_vec_map[callee_addr][4]  # for func level A embed
    func_vec_map[orig_addr] = map_entry
    return (map_entry, ext_lib_fn_names)


def parallelFunc(
    func_chunk,
    addr_to_line,
    enablecfg,
    isUnstripped,
    vocabulary,
    vec_map,
    lock,
    outFile,
    normalize,
    ext_global=None,
    edges_pre=None,
    isCalledd=None,
    mode="",
):

    for func in func_chunk:
        prefix = ""
        prefix = str(func.addr) + "\t"
        ofile = open(outFile, "a")
        if addr_to_line and func.addr in addr_to_line:
            # logic added only for openssl unstr data gen
            src_key = str(addr_to_line[func.addr])
            # Replace config directory in key string for openssl
            src_key = re.sub(
                r"(x86|arm)-(clang|arm-linux-gnueabi-gcc|gcc)-(1?[0-9]|bcf|fla|sub|hybrid)-O[0-3|s]",
                "build-dir",
                src_key,
            )

            prefix += src_key + "\t"
        else:
            prefix += "NO_FILE_SRC \t"

        prefix += func.name + "\t"
        strRefs = []
        if mode != "pre":
            try:
                strEmbed = str_fn_embs[func.addr]
            except KeyError:
                strEmbed = np.array2string(np.zeros(1), max_line_width=99999999)
        else:
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
            strEmbed = getStremb("^".join(strRefs))
        embeddings = emb.symbolicEmbeddings(vocabulary)
        if isUnstripped:
            bbwalkVecs = np.array2string(np.zeros(1), max_line_width=99999999)
        else:

            if args.bb_freq_weight:
                bbwalkVecs_O, bbwalkVecs_T, bbwalkVecs_A = embeddings.processFunc(
                    func, normalize, bbfreq_dict=freq_weight_dict
                )
            else:
                bbwalkVecs_O, bbwalkVecs_T, bbwalkVecs_A = embeddings.processFunc(
                    func, normalize
                )

        try:
            if isUnstripped:
                vec = prefix + str(bbwalkVecs) + "\n"
                lock.acquire()
                ofile.write(vec)
                lock.release()
            # Dump function to file if function is neither a callee nor caller
            # To save on memory usage
            elif func.addr not in isCalledd and not edges_pre[func.addr]:
                if func.addr in ext_global:
                    if mode != "pre":
                        if unstr_df.isin([func.addr]).any().any() and not isUnstripped:
                            # exit(0)
                            unstr_df.loc[
                                unstr_df.addr == func.addr,
                                ["strRefs", "extlibs", "embed_O", "embed_T", "embed_A"],
                            ] = [
                                str(strEmbed).replace("\n", ""),
                                str(ext_global[func.addr]).replace("\n", ""),
                                str(bbwalkVecs_O).replace("\n", ""),
                                str(bbwalkVecs_T).replace("\n", ""),
                                str(bbwalkVecs_A).replace("\n", ""),
                            ]
                    vec = (
                        prefix
                        + str(strEmbed).replace("\n", "")
                        + "\t"
                        + str(ext_global[func.addr]).replace("\n", "")
                        + "\t"
                        + str(bbwalkVecs_O).replace("\n", "")
                        + "\t"
                        + str(bbwalkVecs_T).replace("\n", "")
                        + "\t"
                        + str(bbwalkVecs_A).replace("\n", "")
                        + "\n"
                    )
                else:
                    vec = (
                        prefix
                        + str(strEmbed).replace("\n", "")
                        + "\t"
                        + "\t"
                        + str(bbwalkVecs_O).replace("\n", "")
                        + "\t"
                        + str(bbwalkVecs_T).replace("\n", "")
                        + "\t"
                        + str(bbwalkVecs_A).replace("\n", "")
                        + "\n"
                    )

                lock.acquire()
                ofile.write(vec)
                lock.release()
            else:
                vec_map[func.addr] = [
                    prefix,
                    strEmbed,
                    bbwalkVecs_O,
                    bbwalkVecs_T,
                    bbwalkVecs_A,
                ]
        except KeyError:
            continue


def processBinary(
    proj,
    addr_to_line,
    outFile,
    vocabulary,
    enablecfg=False,
    printTriplets=False,
    isUnstripped=False,
    func_chunks=32,
    dump_cfg=False,
    normalize=1,
    mode="",
):
    if printTriplets:
        ofile = open(outFile, "w")
        triplets = T.triplets()
        # print(triplets)
        print(proj)
        proj.analyses.CFGFast(normalize=True)
        cg = proj.kb.functions.callgraph
        print(cg)
        func_norm = functionNormalizer(proj)
        funcs = []
        cgv = CGV(cg)
        nodes = cgv.sort_nodes()
        traverse = True
        funcs = functionTraversal(
            proj,
            nodes,
            isUnstripped,
            func_norm,
            skipFuncs,
            funcs,
            addr_to_line=addr_to_line,
        )
        # print(funcs)
        # exit(0)
        for i in funcs:
            funcTriplets = triplets.processFunc(i)
            # print(funcTriplets)
            ofile.write(funcTriplets)
        ofile.close()
        exit(0)
    if mode == "pre":
        extern_edges = {}
        edgess = {}
        isCalledd = set()
        ext_lib_fn_names_global = {}
        ext_lib_fn_names_l = {}
        str_fn_embs = {}
        loadFasttextInit()
        symbols = proj.loader.symbols
        ext_lib_functions = [
            symbol.name for symbol in symbols if symbol.is_function and symbol.is_extern
        ]
        while True:
            if psutil.virtual_memory().percent < 45:
                break

        proj.analyses.CFGFast(normalize=True)

    cg = proj.kb.functions.callgraph
    func_norm = functionNormalizer(proj)
    funcs = []
    cgv = CGV(cg)
    nodes = cgv.sort_nodes()
    traverse = True

    funcs = functionTraversal(
        proj,
        nodes,
        isUnstripped,
        func_norm,
        skipFuncs,
        funcs,
        addr_to_line=addr_to_line,
    )
    if mode == "pre":
        func_addr_list = [func.addr for func in funcs]
        processExtlibCalls(
            func_addr_list,
            funcs,
            proj,
            ext_lib_functions,
            ext_lib_fn_names_l,
            extern_edges,
            edgess,
            isCalledd,
        )
        for func in funcs:
            if func.addr in ext_lib_fn_names_l:
                libEmbed = getExtlibemb("^".join(ext_lib_fn_names_l[func.addr]))
            else:
                ext_lib_fn_names_l[func.addr] = np.nan
                libEmbed = getExtlibemb(ext_lib_fn_names_l[func.addr])
            libEmbed = libEmbed.reshape(
                -1,
            )
            ext_lib_fn_names_global[func.addr] = libEmbed

            strRefs = processStringReferences(func, isUnstripped)
            strEmbed = getStremb(strRefs)
            str_fn_embs[func.addr] = strEmbed

    if printTriplets:
        pass
    else:
        ofile = open(outFile, "a")
        embeddings = emb.symbolicEmbeddings(vocabulary)
        if enablecfg:
            first = True
            for func in funcs:
                jsonFnObj = {}
                print("processing function - ", func.name, func.addr)

                jsonFnObj["addr"] = str(func.addr)
                jsonFnObj["name"] = func.name
                if args.dump_dataset and func.addr in addr_to_line:
                    src_key = str(addr_to_line[func.addr])
                    # Replace config directory in key string for openssl
                    src_key = re.sub(
                        r"(x86|arm)-(clang|arm-linux-gnueabi-gcc|gcc)-(1?[0-9]|bcf|fla|sub|hybrid)-O[0-3|s]",
                        "build-dir",
                        src_key,
                    )
                    jsonFnObj["key"] = src_key
                else:
                    jsonFnObj["key"] = "NO_FILE_SRC"

                strRefs = []
                for _, strRef in func.string_references():
                    """
                    preprocess stringrefs for cleaning
                    1. removing everything other than alphabets
                    2. removing strings containing paths
                    3. removing format specifiers
                    4. lowercasing everything
                    5. convert to separate tokens
                    """
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
                sbVecs = embeddings.processFunc(
                    func, enablecfg=enablecfg, genSBVec=True
                )

                json_edges = {}
                for n in proj.kb.functions[func.name].graph:
                    json_edges[n.addr] = [i.addr for i in n.successors()]

                jsonFnObj["nodes"] = sbVecs
                jsonFnObj["edges"] = json_edges

                strVec = strRefs = "^".join(strRefs)
                jsonFnObj["strings"] = strVec

                jsonStr = json.dumps(jsonFnObj)
                if first:
                    ofile.write("[")
                    first = False
                else:
                    ofile.write(",\n")
                ofile.write(jsonStr)

            ofile.write("]\n")
            ofile.close()

        else:
            if dump_cfg == True:
                for func in funcs:
                    G = proj.kb.functions[func.name].graph
                    adj_matrix = nx.adjacency_matrix(G).toarray()
                    if str(adj_matrix.tolist()) == "nan":
                        continue
                    vec = str(func.addr) + "\t" + str(adj_matrix.tolist()) + "\n"
                    ofile.write(vec)

                ofile.write("\n")
                return
                global unstr_df, ext_lib_fn_names
            func_addr_list = [func.addr for func in funcs]
            if not isUnstripped and mode != "pre":
                unstr_df = unstr_df[unstr_df["addr"].isin(func_addr_list)]
                unstr_df.reset_index(drop=True, inplace=True)
            m = Manager()
            lock = m.Lock()
            func_vec_map = m.dict()

            if mode != "pre":
                ext_lib_fn_names_local = ext_lib_fn_names.copy()

            if mode != "pre":
                parallelFunc_part = partial(
                    parallelFunc,
                    addr_to_line=addr_to_line,
                    enablecfg=enablecfg,
                    isUnstripped=isUnstripped,
                    vocabulary=vocabulary,
                    vec_map=func_vec_map,
                    lock=lock,
                    outFile=outFile,
                    normalize=normalize,
                    isCalledd=isCalled,
                    edges_pre=edges,
                    ext_global=ext_lib_fn_names,
                )
            else:
                parallelFunc_part = partial(
                    parallelFunc,
                    addr_to_line=addr_to_line,
                    enablecfg=enablecfg,
                    isUnstripped=isUnstripped,
                    vocabulary=vocabulary,
                    vec_map=func_vec_map,
                    lock=lock,
                    outFile=outFile,
                    normalize=normalize,
                    ext_global=ext_lib_fn_names_global,
                    edges_pre=edgess,
                    isCalledd=isCalledd,
                    mode="pre",
                )
            workers = []
            k = func_chunks
            reversed_func_list = funcs
            reversed_func_list.reverse()
            func_chunks = [
                reversed_func_list[i : i + k]
                for i in range(0, len(reversed_func_list), k)
            ]

            max_jobs_running = 32
            jobs_running = 0
            for chunk in func_chunks:
                while True:
                    if psutil.virtual_memory().percent < 45:
                        break
                p = Process(target=parallelFunc_part, args=(chunk,))
                p.start()
                workers.append(p)

                jobs_running += 1

                if jobs_running >= max_jobs_running:
                    while jobs_running >= max_jobs_running:
                        jobs_running = 0
                        for p in workers:
                            jobs_running += p.is_alive()

            for p in workers:
                p.join()
            p.close()
            if mode != "pre":
                for func in reversed_func_list:
                    if isUnstripped:
                        break
                    try:
                        # Ensure `func.addr` exists in `isCalled` or `edges`
                        if func.addr in isCalled or func.addr in edges:
                            if func.addr not in func_vec_map.keys():
                                continue

                            # Call the addCalleeVecs function
                            map_entry, ext_lib_fn_names_local = addCalleeVecs(
                                func.addr,
                                edges.get(
                                    func.addr, []
                                ),  # Safely get edges for `func.addr`
                                func_vec_map,
                                ext_lib_fn_names_local,
                            )
                            # print("this is map_entry",map_entry)
                            ext_fns = ""
                            if func.addr in ext_lib_fn_names_local:
                                ext_fns = str(ext_lib_fn_names_local[func.addr])
                            else:
                                ext_fns = ""

                            if mode != "pre":
                                if (
                                    unstr_df.isin([func.addr]).any().any()
                                    and not isUnstripped
                                ):
                                    unstr_df.loc[
                                        unstr_df.addr == func.addr,
                                        [
                                            "strRefs",
                                            "extlibs",
                                            "embed_O",
                                            "embed_T",
                                            "embed_A",
                                        ],
                                    ] = [
                                        str(map_entry[1]).replace("\n", ""),
                                        ext_fns.replace("\n", ""),
                                        str(map_entry[2]).replace("\n", ""),
                                        str(map_entry[3]).replace("\n", ""),
                                        str(map_entry[4]).replace("\n", ""),
                                    ]

                            vec = (
                                map_entry[0]
                                + str(map_entry[1]).replace("\n", "")
                                + "\t"
                                + ext_fns.replace("\n", "")
                                + "\t"
                                + str(map_entry[2]).replace("\n", "")
                                + "\t"
                                + str(map_entry[3]).replace("\n", "")
                                + "\t"
                                + str(map_entry[4]).replace("\n", "")
                                + "\n"
                            )
                            ofile.write(vec)
                    except KeyError:
                        continue

                ofile.write("\n")
                ofile.close()

                if not isUnstripped and mode != "pre":
                    unstr_df = unstr_df[
                        unstr_df["embed_T"].notnull()
                    ]  # Use `notnull()` for better pandas practice
                    unstr_df.reset_index(drop=True, inplace=True)

                return
            else:
                for func in reversed_func_list:
                    if isUnstripped:
                        break
                    try:
                        # Ensure `func.addr` exists in `isCalled` or `edges`
                        if func.addr in isCalledd or func.addr in edgess:
                            if func.addr not in func_vec_map.keys():
                                continue

                            # Call the addCalleeVecs function
                            map_entry, ext_lib_fn_names_global = addCalleeVecs(
                                func.addr,
                                edgess.get(
                                    func.addr, []
                                ),  # Safely get edges for `func.addr`
                                func_vec_map,
                                ext_lib_fn_names_global,
                            )

                            ext_fns = ""
                            if func.addr in ext_lib_fn_names_global:
                                ext_fns = str(ext_lib_fn_names_global[func.addr])
                            else:
                                ext_fns = ""
                            vec = (
                                map_entry[0]
                                + str(map_entry[1]).replace("\n", "")
                                + "\t"
                                + ext_fns.replace("\n", "")
                                + "\t"
                                + str(map_entry[2]).replace("\n", "")
                                + "\t"
                                + str(map_entry[3]).replace("\n", "")
                                + "\t"
                                + str(map_entry[4]).replace("\n", "")
                                + "\n"
                            )
                            ofile.write(vec)
                    except KeyError:
                        continue

                ofile.write("\n")
                ofile.close()

                if not isUnstripped and mode != "pre":
                    unstr_df = unstr_df[
                        unstr_df["embed_T"].notnull()
                    ]  # Use `notnull()` for better pandas practice
                    unstr_df.reset_index(drop=True, inplace=True)

                return


def loadBinary(binary, args):
    if args.mode == "db":
        print("\033[31mDB Flow Activated\033[0m")
        global str_fn_embs, ext_lib_fn_names, isCalled, edges, cgv_nodes
        isUnstripped = False
        if "unstripped" in binary:
            isUnstripped = True
        META_DB_PATH = os.path.abspath(args.embedding_db_path)
        engine = db.create_engine(f"sqlite:///{META_DB_PATH}")
        session = sessionmaker(bind=engine)
        connection = session()
        # print("I have gone here")
        # print(args.dump_dataset)
        # exit(0)
        if args.dump_dataset:
            binary_db = binary.replace("stripped", DB_PATH).replace(".out", ".db")
            # print(binary_db)
            # exit(0)
            proj_db = AngrDB()
            proj = proj_db.load(binary_db)
            db_bin_key = "_".join(binary.split("/")[-4:])
            if isUnstripped:
                result = (
                    connection.query(binaryMetaDataUnstripped)
                    .filter(binaryMetaDataUnstripped.id == db_bin_key)
                    .all()
                )
                # print(result)
                addr_to_line = pickle.loads(result[0].addr_to_line)
                cgv_nodes = pickle.loads(result[0].cgv_nodes)
            else:
                result = (
                    connection.query(binaryMetaData)
                    .filter(binaryMetaData.id == db_bin_key)
                    .all()
                )
                ext_lib_fn_names = pickle.loads(result[0].ext_lib_functions)
                str_fn_embs = pickle.loads(result[0].string_func_embedding)
                isCalled = pickle.loads(result[0].isCalled)
                edges = pickle.loads(result[0].func_callee_edges)
                addr_to_line = None
                cgv_nodes = pickle.loads(result[0].cgv_nodes)
        else:
            binary_db = binary.replace("stripped", DB_PATH).replace(".out", ".db")
            proj_db = AngrDB()
            proj = proj_db.load(binary_db)

        if args.output:
            outFile = args.output
        elif args.output_dir_name:
            if args.dump_freq:
                outputDir = os.path.dirname(binary).replace(
                    "Datafiles", args.output_dir_name
                )
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir, exist_ok=True)
                outFile = os.path.join(
                    outputDir, os.path.basename(binary)[:-4] + ".pickle"
                )
            else:
                outputDir = os.path.join(
                    os.path.dirname(os.path.dirname(binary)), args.output_dir_name
                )
                if not os.path.exists(outputDir):
                    os.mkdir(outputDir)
                outFile = os.path.join(
                    outputDir, os.path.basename(binary)[:-4] + ".data"
                )

        if os.path.exists(outFile):
            print(outFile, " already exists")
            return
        if not isUnstripped:
            unstr_file = os.path.abspath(outFile.replace("stripped", "unstripped"))
            if not os.path.exists(unstr_file):
                print("Unstripped Counter Part not present, Generate it first. Exiting")
                exit(0)

            global unstr_df
            col_names_unstr = ["addr", "key", "fnName", "embed_unst"]
            unstr_df = pd.read_csv(
                unstr_file, sep="\t", names=col_names_unstr, header=None
            )
            unstr_df = unstr_df[unstr_df["key"].str.contains("NO_FILE_SRC") == False]
            unstr_df.reset_index(drop=True, inplace=True)
            unstr_df = unstr_df[unstr_df["fnName"].str.startswith("sub_") == False]
            unstr_df.reset_index(drop=True, inplace=True)
            unstr_df.drop("embed_unst", axis=1, inplace=True)
            unstr_df.key = unstr_df.key + unstr_df.fnName
            unstr_df.drop("fnName", axis=1, inplace=True)
            unstr_df = unstr_df.assign(
                strRefs=None, extlibs=None, embed_O=None, embed_T=None, embed_A=None
            )

        print("Processing", binary)
        if args.dump_dataset:
            status = processBinary(
                proj,
                addr_to_line,
                outFile,
                args.vocabulary,
                args.enable_cfg,
                args.dump_triplets,
                isUnstripped,
                args.func_chunks,
                args.dump_cfg,
                args.normalize,
            )
        else:
            status = processBinary(
                proj,
                None,
                outFile,
                args.vocabulary,
                args.enable_cfg,
                args.dump_triplets,
                isUnstripped,
                args.func_chunks,
                args.dump_cfg,
                args.normalize,
            )

        if status == -1:
            print("Angr Error:", binary)

        if not isUnstripped:
            unstr_df.to_csv(outFile.replace(".data", ".csv"), sep="\t")

    if args.mode == "non-db":
        print("\033[31mNon-DB Flow Activated\033[0m")
        if args.dump_dataset:
            proj = angr.Project(binary, auto_load_libs=False, load_debug_info=True)
            addr_to_line = proj.loader.main_object.addr_to_line
        else:
            proj = angr.Project(binary, auto_load_libs=False)

        symbols = proj.loader.symbols
        isUnstripped = False
        if "unstripped" in binary:
            isUnstripped = True

        if args.output:
            outFile = args.output
        elif args.output_dir_name:
            if args.dump_freq:
                outputDir = os.path.dirname(binary).replace(
                    "Datafiles", args.output_dir_name
                )
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir, exist_ok=True)
                outFile = os.path.join(
                    outputDir, os.path.basename(binary)[:-4] + ".pickle"
                )
            else:
                outputDir = os.path.join(
                    os.path.dirname(os.path.dirname(binary)), args.output_dir_name
                )
                if not os.path.exists(outputDir):
                    os.mkdir(outputDir)
                outFile = os.path.join(
                    outputDir, os.path.basename(binary)[:-4] + ".data"
                )

        if os.path.exists(outFile):
            print(outFile, " already exists")
            return
        print("Processing", binary)
        if args.dump_dataset:
            status = processBinary(
                proj,
                addr_to_line,
                outFile,
                args.vocabulary,
                args.enable_cfg,
                args.dump_triplets,
                isUnstripped,
                args.func_chunks,
                args.dump_cfg,
                args.normalize,
                "pre",
            )
        else:
            status = processBinary(
                proj,
                None,
                outFile,
                args.vocabulary,
                args.enable_cfg,
                args.dump_triplets,
                isUnstripped,
                args.func_chunks,
                args.dump_cfg,
                args.normalize,
                "pre",
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Tool to generate VEXIR2Vec representations"
    )
    group_input = parser.add_mutually_exclusive_group(required=True)
    group_output = parser.add_mutually_exclusive_group(required=True)
    group_input.add_argument("-b", "--binary", type=str, help="Input binary")
    group_input.add_argument(
        "-b_dir",
        "--binary_dir",
        type=str,
        help="Directory containing input binaries or subdirectories of input binaries",
    )
    parser.add_argument(
        "-threads",
        "--num_threads",
        type=int,
        default=1,
        help="Number of parallel Pool workers",
    )
    parser.add_argument(
        "-fchunks",
        "--func_chunks",
        type=int,
        default=32,
        help="Number of function chunks to be processed in parallel",
    )
    group_output.add_argument("-o", "--output", type=str, help="Output file path")
    group_output.add_argument(
        "-o_dir", "--output_dir_name", type=str, help="Output directory name"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t",
        "--dump-triplets",
        dest="dump_triplets",
        required=False,
        default=False,
        action="store_true",
        help="Print triplets to the file",
    )
    group.add_argument("-v", "--vocabulary", help="path to the vocabulary")
    parser.add_argument(
        "-d",
        "--dump-dataset",
        dest="dump_dataset",
        required=False,
        default=False,
        action="store_true",
        help="Print dataset to the file",
    )
    group_freq = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument(
        "-cfg",
        "--enable-cfg",
        dest="enable_cfg",
        required=False,
        default=False,
        action="store_true",
        help="Generate data with cfg edges",
    )
    parser.add_argument(
        "-dc",
        "--dump-cfg",
        dest="dump_cfg",
        required=False,
        default=False,
        action="store_true",
        help="Dump only fn address and adj matrix coresponding to its cfg",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        type=int,
        default=1,
        help="The normalisation",
    )
    parser.add_argument(
        "-w",
        "--bb-freq-weight",
        dest="bb_freq_weight",
        required=False,
        help="Path to frequency json to add weights to embeddings",
    )
    parser.add_argument(
        "-edb",
        "--embedding-db",
        dest="embedding_db_path",
        type=str,
        required=False,
        help="Path to DB containing extlib and string embeddings",
    )
    parser.add_argument(
        "-mode",
        dest="mode",
        type=str,
        required=False,
        help="Specify the mode as a string",
    )

    args = parser.parse_args()
    if args.bb_freq_weight:
        with open(args.bb_freq_weight, "rb") as f:
            freq_weight_dict = pickle.load(f)

    if args.binary:
        loadBinary(os.path.abspath(args.binary), args)
    elif args.binary_dir:
        binaries = glob.glob(args.binary_dir + "/*/unstripped/*.out")
        binaries_args = [(os.path.abspath(binary), args) for binary in binaries]
        with ProcessPool(args.num_threads) as p:
            result = p.map(loadBinary, binaries_args, chunksize=1, timeout=7200)
