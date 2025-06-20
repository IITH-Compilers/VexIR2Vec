# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Test Set Generation Script for Searching Experiment"""

# Usage: python3 test-set-gen-nolib.py -xd $x86_DATA -ad $ARM_DATA -p $PROJ -o $OUTPUT_DIR
# Ex: python3 test-set-gen-nolib.py -xd /path/to/x86-data-files/ -ad /path/to/arm-data-files/ -o /path/to/output-groundtruth/
# Desc: Generates a directory for the project, containing ground truth csv for each experiment config, arranged experiment wise


import numpy as np
import pandas as pd
import argparse
import os
import sys
import re

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

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

# psusan missing for arm-clang-12-O2
test_dict_missing = {
    "putty": [
        "cgtest",
        "fuzzterm",
        "osxlaunch",
        "plink",
        "pscp",
        "psftp",
        "psocks",
        "puttygen",
        "testcrypt",
        "testsc",
        "testzlib",
        "uppity",
    ]
}

configs = [
    "x86-clang-12-O0",
    "x86-clang-12-O1",
    "x86-clang-12-O2",
    "x86-clang-12-O3",
    "x86-clang-8-O0",
    "x86-clang-8-O1",
    "x86-clang-8-O2",
    "x86-clang-8-O3",
    "x86-clang-6-O0",
    "x86-clang-6-O1",
    "x86-clang-6-O2",
    "x86-clang-6-O3",
    "x86-gcc-10-O0",
    "x86-gcc-10-O1",
    "x86-gcc-10-O2",
    "x86-gcc-10-O3",
    "x86-gcc-8-O0",
    "x86-gcc-8-O1",
    "x86-gcc-8-O2",
    "x86-gcc-8-O3",
    "x86-gcc-6-O0",
    "x86-gcc-6-O1",
    "x86-gcc-6-O2",
    "x86-gcc-6-O3",
]


def modifyKey(x):

    x = re.split(", |\)", x)
    modified_str = x[0] + ")" + x[2]
    return modified_str


def loadFiles(path, config):
    for proj in ["coreutils", "diffutils", "findutils", "curl", "gzip", "lua", "putty"]:
        proj_path = os.path.join(path, proj)
        unstripped_path = os.path.join(
            proj_path, config, "/path/to/unstripped-data-files/"
        )
        stripped_path = os.path.join(proj_path, config, "/path/to/stripped-data-files/")
        filenames = os.listdir(unstripped_path)
        if config == "arm-clang-12-O2" and proj == "putty":
            test_list = test_dict_missing[proj]
        else:
            test_list = test_dict[proj]
        test_list = [bin + ".data" for bin in test_list]
        for filename in filenames:
            if filename in test_list:
                unstripped_df = pd.read_csv(
                    os.path.join(unstripped_path, filename),
                    sep="\t",
                    header=None,
                    names=["addr", "key", "name", "embed"],
                )
                stripped_df = pd.read_csv(
                    os.path.join(stripped_path, filename),
                    sep="\t",
                    header=None,
                    names=[
                        "addr",
                        "key",
                        "name",
                        "str",
                        "lib",
                        "embed_O",
                        "embed_T",
                        "embed_A",
                    ],
                )
                stripped_df = stripped_df.astype(
                    {"embed_O": str, "embed_T": str, "embed_A": str}
                )
                stripped_df.drop(["key"], axis=1, inplace=True)
                unstripped_df.drop(["embed"], axis=1, inplace=True)

                unstripped_df = pd.merge(
                    unstripped_df, stripped_df, on="addr", suffixes=("_x", "_y")
                )
                unstripped_df["bin_name"] = filename.split(".data")[0]
                unstripped_df["proj"] = proj
                print(unstripped_df.columns)
                print(stripped_df.columns)
                yield unstripped_df


def loadData(path, config):
    data = pd.concat(loadFiles(path, config))
    data = (
        data[data["key"].str.contains("NO_FILE_SRC") == False]
        .reset_index()
        .drop(["index"], axis=1)
        .astype({"addr": str})
    )
    data = (
        data[data["name_x"].str.startswith("sub_") == False]
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

    data["key"] = data["key"] + data["name_x"]
    data = data.drop(["name_x", "name_y"], axis=1)
    data.drop_duplicates(inplace=True)
    data.key = data.key.apply(modifyKey)
    data.rename({"addr": "address", "embed": "emb"}, axis=1, inplace=True)

    return data


def generateSet(dir_path, config, out_path):
    data = loadData(dir_path, config)
    print(data.shape)
    suffix = "_" + config
    testset = data
    testset = testset.reset_index()
    testset = testset.drop(["index"], axis=1)
    testset = testset.loc[
        :,
        [
            "proj",
            "bin_name",
            "key",
            "address",
            "str",
            "lib",
            "embed_O",
            "embed_T",
            "embed_A",
        ],
    ]
    test_correct = testset
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok=True)
    test_correct.rename(
        {
            "bin_name": "bin_name" + suffix,
            "key": "key" + suffix,
            "address": "address" + suffix,
            "str": "str" + suffix,
            "lib": "lib" + suffix,
            "embed_O": "embed_O" + suffix,
            "embed_T": "embed_T" + suffix,
            "embed_A": "embed_A" + suffix,
        },
        axis=1,
        inplace=True,
    )
    test_correct = test_correct[
        [
            "proj",
            "bin_name" + suffix,
            "key" + suffix,
            "address" + suffix,
            "str" + suffix,
            "lib" + suffix,
            "embed_O" + suffix,
            "embed_T" + suffix,
            "embed_A" + suffix,
        ]
    ]
    test_correct.to_csv(
        os.path.join(out_path, "binsearch-" + config + ".csv"), index=False
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-xd",
        "--x86_data_path",
        dest="x86_data_path",
        help="x86 data path",
        default=None,
    )
    parser.add_argument(
        "-ad",
        "--arm_data_path",
        dest="arm_data_path",
        help="ARM data path",
        default=None,
    )
    parser.add_argument(
        "-o", "--out_path", dest="out_path", help="Output path", default=None
    )
    parser.add_argument(
        "-n", "--config_num", dest="config_num", help="Index to config", default=None
    )
    args = parser.parse_args()

    base_out_path = args.out_path
    n = int(args.config_num)
    print(n)
    x86_data_path = args.x86_data_path
    arm_data_path = args.arm_data_path

    if n in range(0, 4):
        generateSet(
            x86_data_path, configs[n], os.path.join(base_out_path, "x86-clang-12")
        )
    elif n in range(4, 8):
        generateSet(
            x86_data_path, configs[n], os.path.join(base_out_path, "x86-clang-8")
        )
    elif n in range(8, 12):
        generateSet(
            x86_data_path, configs[n], os.path.join(base_out_path, "x86-clang-6")
        )
    elif n in range(12, 16):
        generateSet(
            x86_data_path, configs[n], os.path.join(base_out_path, "x86-gcc-10")
        )
    elif n in range(16, 20):
        generateSet(x86_data_path, configs[n], os.path.join(base_out_path, "x86-gcc-8"))
    elif n in range(20, 24):
        generateSet(x86_data_path, configs[n], os.path.join(base_out_path, "x86-gcc-6"))
    else:
        print("Invalid number: ", n)
