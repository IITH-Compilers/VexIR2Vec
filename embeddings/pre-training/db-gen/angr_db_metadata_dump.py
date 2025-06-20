# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""This script is used to create the database files"""

# Usage:
# - Running in single binary mode:
# python angr_db_metadata_dump.py -b <binary_path> -adb <angr_db_path> -edb <embedding_db_path> -t <number_of_threads>

# - Running in directory mode:
# python angr_db_metadata_dump.py -b_dir <directory_path> -adb <angr_db_path> -edb <embedding_db_path> -t <number_of_threads>

from time import sleep
import angr
import sys
from angr.angrdb.db import AngrDB
import os
from binary_db_model import *

SCRIPT_DIR_1 = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR = os.path.join(SCRIPT_DIR_1, "../embedding-gen")
sys.path.append(os.path.normpath(IMPORT_DIR))
print(IMPORT_DIR)
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import argparse
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError, NoResultFound
from angr.analyses.forward_analysis.visitors.call_graph import CallGraphVisitor as CGV
from function_normalizer import functionNormalizer
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import sys
import string
import psutil
import re
import pickle
import glob

sys.path.append("../../vexNet/")
from utils import *
import logging
import threading
import time
import signal

logging.getLogger("angr").setLevel(logging.CRITICAL)

# Load the fasttext model once.

angr_db_path = None
embedding_db_path = None
isAngrDb = True
isEmbDb = True
isUnstripped = False
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


def startThreadToTerminateWhenParentProcessDies(ppid):
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def removeEntity(text, entity_list):
    for entity in entity_list:
        if entity in text:
            text = text.replace(entity, " ")
    return text.strip()


def checkPaths(args):
    if args.binary_dir:
        binary_path = os.path.abspath(args.binary_dir)
        if not os.path.exists(binary_path):
            print("Binary File not Found")
            exit(0)
    elif args.binary:
        binary_path = os.path.abspath(args.binary)
        if not os.path.exists(binary_path):
            print("Binary File not Found")
            exit(0)

    angr_db_path = os.path.abspath(args.angr_db_path)
    if not os.path.exists(angr_db_path):
        print("Invalid Angr DB path")
        exit(0)

    embedding_db_path = os.path.abspath(args.embedding_db_path)
    if not os.path.exists(os.path.dirname(embedding_db_path)):
        print("Invalid Embedding DB Path")
        exit(0)
    return binary_path, angr_db_path, embedding_db_path


def checkDbEntry(angr_db_path, embedding_db_path, binary_path, db_session):
    global isAngrDb, isEmbDb

    bin_id = "_".join(binary_path.split("/")[-4:])
    try:
        if "unstripped" in binary_path:
            startDbUnstripped(embedding_db_path)
            result = (
                db_session.query(binaryMetaDataUnstripped)
                .filter(binaryMetaDataUnstripped.id == bin_id)
                .one()
            )
            del result
        else:
            startDb(embedding_db_path)
            result = (
                db_session.query(binaryMetaData)
                .filter(binaryMetaData.id == bin_id)
                .one()
            )
            del result
    except NoResultFound:
        isEmbDb = True
    except OperationalError:
        pass
    if not os.path.exists(angr_db_path):
        isAngrDb = True
    else:
        tmp_db = AngrDB()
        try:
            tmp_proj = tmp_db.load(angr_db_path)
            del tmp_proj
        except Exception:
            isAngrDb = True
        del tmp_db
    return bin_id


def generateDataStripped(binary_path, angr_db_path, bin_id, isAngrDb=True):
    extern_edges = {}
    edges = {}
    isCalled = set()
    ext_lib_fn_names_global = {}
    ext_lib_fn_names = {}
    str_fn_embs = {}
    # startDb(embedding_db_path)
    loadFasttextInit()
    print("Fasttest Model loaded")
    project = angr.Project(binary_path, auto_load_libs=False, load_debug_info=True)

    addr_to_line = project.loader.main_object.addr_to_line
    symbols = project.loader.symbols
    ext_lib_functions = [
        symbol.name for symbol in symbols if symbol.is_function and symbol.is_extern
    ]

    # Check my memory usage - Safety check
    while True:
        if psutil.virtual_memory().percent < 45:
            break

    project.analyses.CFGFast(normalize=True)
    func_norm = functionNormalizer(project)
    funcs = []
    cg = project.kb.functions.callgraph
    cgv = CGV(cg)
    nodes = cgv.sort_nodes()

    funcs = functionTraversal(
        project,
        nodes,
        isUnstripped,
        func_norm,
        skipFuncs,
        funcs,
        addr_to_line=addr_to_line,
    )
    func_addr_list = [func.addr for func in funcs]
    processExtlibCalls(
        func_addr_list,
        funcs,
        project,
        ext_lib_functions,
        ext_lib_fn_names,
        extern_edges,
        edges,
        isCalled,
    )
    for func in funcs:
        if func.addr in ext_lib_fn_names:
            libEmbed = getExtlibemb("^".join(ext_lib_fn_names[func.addr]))
        else:
            ext_lib_fn_names[func.addr] = np.nan
            libEmbed = getExtlibemb(ext_lib_fn_names[func.addr])
        libEmbed = libEmbed.reshape(
            -1,
        )
        ext_lib_fn_names_global[func.addr] = libEmbed

        strRefs = processStringReferences(func, isUnstripped)
        strEmbed = getStremb(strRefs)
        str_fn_embs[func.addr] = strEmbed

    if isAngrDb:
        angr_db = AngrDB(project)
        angr_db.dump(angr_db_path)
        print(f"Dumped Angr Project: {os .path .basename (binary_path )}")
        del angr_db

    ext_lib_pkl = pickle.dumps(ext_lib_fn_names_global)
    str_emb_pkl = pickle.dumps(str_fn_embs)
    isCalled_pkl = pickle.dumps(isCalled)
    edges_pkl = pickle.dumps(edges)
    cgv_nodes = pickle.dumps(nodes)
    bin_meta_data = binaryMetaData(
        bin_id, ext_lib_pkl, str_emb_pkl, isCalled_pkl, edges_pkl, cgv_nodes
    )
    return bin_meta_data


def generateDataUnstripped(binary_path, angr_db_path, bin_id, isAngrDb=True):
    project = angr.Project(binary_path, auto_load_libs=False, load_debug_info=True)
    addr_to_line = project.loader.main_object.addr_to_line
    project.analyses.CFGFast(normalize=True)
    cg = project.kb.functions.callgraph
    cgv = CGV(cg)
    nodes = cgv.sort_nodes()
    if isAngrDb:
        angr_db = AngrDB(project)
        angr_db.dump(angr_db_path)
        print(f"Dumped Angr Project Unstripped: {os .path .basename (binary_path )}")
        del angr_db

    addr_to_line = pickle.dumps(addr_to_line)
    cgv_nodes = pickle.dumps(nodes)
    bin_meta_data = binaryMetaDataUnstripped(bin_id, addr_to_line, cgv_nodes)
    return bin_meta_data

    # ===========================================DRIVER CODE=====================================


parser = argparse.ArgumentParser(description="Tool to Dump Binary Metadata in DB")
parser.add_argument("-b", "--binary", type=str, help="Input binary")
parser.add_argument(
    "-b_dir",
    "--binary_dir",
    type=str,
    help="Directory containing input binaries or subdirectories of input binaries",
)
parser.add_argument(
    "-adb", "--angr_db_path", type=str, help="Path to store Angr Proj db"
)
parser.add_argument(
    "-edb", "--embedding_db_path", type=str, help="Path to store embedding db"
)
parser.add_argument(
    "-t", "--threads", type=int, default=5, help="Numbers of Process to spawn"
)
args = parser.parse_args()

binary_path, angr_db_path, embedding_db_path = checkPaths(args)
if "unstripped" in binary_path:
    isUnstripped = True
    startDbUnstripped(embedding_db_path)
else:
    startDb(embedding_db_path)

engine = create_engine(
    f"sqlite:///{embedding_db_path }",
    pool_size=10,
    max_overflow=20,
    connect_args={"timeout": 30},
)
db_session = scoped_session(sessionmaker(bind=engine))

if args.binary_dir:
    binary_arg_tup = []
    binaries = glob.glob(
        args.binary_dir + "/*.out"
    )  # change stripped=>unstripped for generating data files for unstripped binaries
    for binary in binaries:
        bin_angr_path = (
            angr_db_path + "/" + os.path.basename(binary).replace(".out", ".db")
        )

        bin_id = "_".join(binary.split("/")[-4:])
        if not isAngrDb and not isEmbDb:
            db_session.close()
            engine.dispose()
            print("DB's present, no need of generation", isAngrDb, isEmbDb)
            continue

        binary_arg_tup.append((binary, bin_angr_path, bin_id, isAngrDb))
        isAngrDb = True
        isEmbDb = True

    with ProcessPoolExecutor(
        max_workers=args.threads,
        initializer=startThreadToTerminateWhenParentProcessDies,
        initargs=(os.getpid(),),
    ) as executor:
        if isUnstripped:
            futures = [
                executor.submit(generateDataUnstripped, *args)
                for args in binary_arg_tup
            ]
        else:
            futures = [
                executor.submit(generateDataStripped, *args) for args in binary_arg_tup
            ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print("Result: ", result.id)
            while True:
                try:
                    db_session.add(result)
                    db_session.commit()
                    print("Data Commited Successfully")
                    break
                except IntegrityError as err:
                    print("Data already present: ", err)
                    db_session.rollback()
                    sleep(0.01)
                    break
                except OperationalError as err:
                    print("Fatal DB Error, Kindly Check DB: ", err)
                    db_session.rollback()
                    sleep(0.01)

if args.binary:
    angr_db_path += "/" + os.path.basename(args.binary).replace(".out", ".db")
    bin_id = checkDbEntry(angr_db_path, embedding_db_path, args.binary, db_session)
    if isUnstripped:
        result = generateDataUnstripped(args.binary, angr_db_path, bin_id, isAngrDb)
    else:
        result = generateDataStripped(args.binary, angr_db_path, bin_id, isAngrDb)
    while True:
        try:
            db_session.add(result)
            db_session.commit()
            print("Data Commited Successfully")
            break
        except IntegrityError as err:
            print("Data already present: ", err)
            break
        except OperationalError as err:
            print("Fatal DB Error, Kindly Check DB: ", err)
            db_session.rollback()
            sleep(0.01)


db_session.close()
engine.dispose()
