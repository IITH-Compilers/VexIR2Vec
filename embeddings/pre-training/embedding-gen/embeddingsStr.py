# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec String Embedding Generation Script"""

from functools import lru_cache
import re
from re import search
import string
import argparse
import angr
import pyvex as py
import numpy as np
import fasttext
import fasttext.util
import embeddings as emb
from angr.analyses.forward_analysis.visitors.call_graph import CallGraphVisitor as CGV


@lru_cache
def getModel():
    ft = fasttext.load_model("./vexIR/cc.en.300.bin")
    fasttext.util.reduce_model(ft, 100)
    return ft


def removeEntity(text, entity_list):
    for entity in entity_list:
        if entity in text:
            text = text.replace(entity, " ")
    return text.strip()


def getEmbedding(strRefs):
    ft = getModel()
    vectors = [ft.get_word_vector(word).reshape((1, 100)) for word in strRefs]
    return np.sum(np.concatenate(vectors, axis=0), axis=0)


def collectStringRefs(args):
    cfg = proj.analyses.CFGFast()
    cg = cfg.functions.callgraph
    funcs = []
    cgv = CGV(cg)
    nodes = cgv.sort_nodes()
    traverse = True
    for fn_addr in nodes:
        fnName = cfg.functions.function(fn_addr).name
        if traverse:
            if cfg.kb.functions.function(fn_addr) is None:
                continue
            else:
                funcs.append(cfg.kb.functions[fnName])

    ofile = open(outFile, "w")
    # embeddings = emb.SymbolicEmbeddings(args.vocabulary)
    for func in funcs:
        print("processing function - ", func.name, func.addr)
        strRefs = []
        prefix = ""
        prefix = str(func.addr) + "\t"
        if func.addr in addr_to_line:
            prefix += str(addr_to_line[func.addr]) + "\t"
        else:
            prefix += "NO_FILE_SRC \t"
        prefix += func.name + "\t"

        for _, strRef in func.string_references(vex_only=True):
            """
            preprocess stringrefs for cleaning
            1. removing everything other than alphabets
            2. removing strings containing paths
            3. removing format specifiers
            4. lowercasing everything
            5. convert to separate tokens
            """
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
            # print('strRef: ', strRef)
            if strRef:
                strRefs.extend(strRef)

        funcVec = getEmbedding(strRefs) if strRefs else np.array([0])
        # sbVecs = embeddings.processFunc(i, genSBVec=True)
        vec = prefix + str(funcVec) + "\n"
        ofile.write(vec)

    ofile.close()

    # Code below for the multiprocessing of binaries for generation of sbVecs and string embeddings
    # @lru_cache decorator may help for loading


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to generate VEXIR2Vec representations"
    )
    parser.add_argument("-b", "--binary", type=str, help="Input binary")
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output file path"
    )
    parser.add_argument("-v", "--vocabulary", help="path to the vocabulary")
    args = parser.parse_args()

    proj = angr.Project(args.binary, auto_load_libs=False, load_debug_info=True)
    addr_to_line = proj.loader.main_object.addr_to_line

    outFile = args.output
    collectStringRefs(args)
