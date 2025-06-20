# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec OOV Script for collecting OOV Counts"""

from posixpath import basename
import triplets as T
import embeddings_collect_oov_count as emb
import argparse
import angr
import pyvex as py
import numpy as np
from angr.analyses.forward_analysis.visitors.call_graph import CallGraphVisitor as CGV

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


def main(bin_path, outFile, vocabulary, printTriplets=False):
    tStr = ""
    cfg = proj.analyses.CFGFast()
    cg = cfg.functions.callgraph
    node_addrs = [addr for addr in cg.nodes]
    funcs = []
    cgv = CGV(cg)
    nodes = cgv.sort_nodes()
    traverse = True
    for fn_addr in nodes:
        fnName = cfg.functions.function(fn_addr).name
        if traverse:
            if cfg.kb.functions.function(fn_addr) is None:
                print("skipping - ", fnName)
                continue
            else:
                funcs.append(cfg.kb.functions[fnName])

    ofile = open(outFile, "a")
    unk_ent_ofile = open(outFile.split(".")[0] + "-unk-ent.txt", "a")
    if printTriplets:
        triplets = T.Triplets()
        for i in funcs:
            funcTriplets = triplets.processFunc(i)
    else:
        embeddings = emb.SymbolicEmbeddings(vocabulary)
        for i in funcs:

            funcVec = np.zeros(embeddings.dim)
            prefix = ""
            funcVec = embeddings.processFunc(i)

            # Print missing opcodes
        unk_key_count_str = ""
        for key in embeddings.unk_key_dict.keys():
            unk_key_count_str += " " + key + ": " + str(embeddings.unk_key_dict[key])
        unk_ent_ofile.write(
            bin_path + ": " + str(embeddings.unknown_count) + unk_key_count_str + "\n"
        )
        ofile.write(str(embeddings.unknown_count) + "\n")
    ofile.close()
    unk_ent_ofile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to generate VEXIR2Vec representations"
    )
    parser.add_argument("-b", "--binary", type=str, help="Input binary")
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output file path"
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
    args = parser.parse_args()

    if args.dump_dataset:
        proj = angr.Project(args.binary, auto_load_libs=False, load_debug_info=True)
        addr_to_line = proj.loader.main_object.addr_to_line
    else:
        proj = angr.Project(args.binary)

    outFile = args.output

    main(args.binary, outFile, args.vocabulary, args.dump_triplets)
