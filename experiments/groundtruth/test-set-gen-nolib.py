# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Test Set Generation Script for Diffing Experiment"""

# Usage: python3 test-set-gen-nolib.py -xd $x86_DATA -ad $ARM_DATA -p $PROJ -o $OUTPUT_DIR
# Ex: python3 test-set-gen-nolib.py -xd /path/to/x86-data-all -ad /path/to/arm32-data-all -o /path/to/dataset
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


def modifyKey(x):
    x = re.split(", |\)", x)
    modified_str = x[0] + ")" + x[2]
    return modified_str


def loadFiles(path, filenames, test_list):
    stripped_path = os.path.join(
        path.rsplit("/", 1)[0], "/path/to/stripped-data-files/"
    )
    for filename in filenames:
        if filename in test_list:
            unstripped_df = pd.read_csv(
                os.path.join(path, filename),
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
            stripped_df.drop(["key", "str", "lib"], axis=1, inplace=True)

            unstripped_df.drop(["embed"], axis=1, inplace=True)

            unstripped_df = pd.merge(
                unstripped_df, stripped_df, on="addr", suffixes=("_x", "_y")
            )
            unstripped_df["bin_name"] = filename.split(".data")[0]
            print(unstripped_df.columns)
            print(stripped_df.columns)
            yield unstripped_df


def loadData(path, test_list):
    data = pd.concat(loadFiles(path, os.listdir(path), test_list))
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
    data.drop(["embed_O", "embed_T", "embed_A"], axis=1, inplace=True)
    data["embed"] = "[]"

    data["key"] = data["key"] + data["name_x"]
    data = data.drop(["name_x", "name_y"], axis=1)
    data.drop_duplicates(inplace=True)
    data.key = data.key.apply(modifyKey)
    data.rename({"addr": "address", "embed": "emb"}, axis=1, inplace=True)

    return data


def generateSet(dir_path_x, dir_path_y, config, test_list, out_path):
    config_x = config[0]
    config_y = config[1]
    src = os.path.join(
        dir_path_x, os.path.join(config_x, "/path/to/unstripped-data-files/")
    )
    tgt = os.path.join(
        dir_path_y, os.path.join(config_y, "/path/to/unstripped-data-files/")
    )
    print(src)
    print(tgt)
    # 1
    srcData = loadData(src, test_list)
    tgtData = loadData(tgt, test_list)
    print(srcData.shape)
    print(tgtData.shape)
    # 2
    suffix_x = "_" + config_x
    suffix_y = "_" + config_y
    data = pd.merge(
        srcData, tgtData, on=["key", "bin_name"], suffixes=(suffix_x, suffix_y)
    )
    print(data.shape)
    # 3
    print(data.shape)
    testset = data
    testset = testset.reset_index()
    testset = testset.drop(["index"], axis=1)
    testset = testset.loc[
        :,
        [
            "key",
            "bin_name",
            "address" + suffix_x,
            "emb" + suffix_x,
            "address" + suffix_y,
            "emb" + suffix_y,
        ],
    ]
    test_correct = testset
    proj_name = os.path.basename(dir_path_x)
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok=True)
    test_correct.rename({"key": "key" + suffix_x}, axis=1, inplace=True)
    test_correct.rename({"bin_name": "bin_name" + suffix_x}, axis=1, inplace=True)
    test_correct["key" + suffix_y] = test_correct["key" + suffix_x]
    test_correct["bin_name" + suffix_y] = test_correct["bin_name" + suffix_x]
    test_correct = test_correct[
        [
            "key" + suffix_x,
            "key" + suffix_y,
            "bin_name" + suffix_x,
            "address" + suffix_x,
            "bin_name" + suffix_y,
            "address" + suffix_y,
            "emb" + suffix_x,
            "emb" + suffix_y,
        ]
    ]
    test_correct.to_csv(
        os.path.join(
            out_path, proj_name + "-full-" + config_x + "-" + config_y + ".csv"
        ),
        index=False,
    )


def generateSets(dir_path_x, dir_path_y, configs, test_list, out_path):
    print(len(test_list))
    print(test_list)
    for config in configs:
        config_x = config[0]
        config_y = config[1]
        print("Config:")
        print(config_x)
        print(config_y)
        generateSet(dir_path_x, dir_path_y, config_x, config_y, test_list, out_path)


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

test_dict_missing = {
    "putty": [
        "cgtest",
        "fuzzterm",
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
    ("arm-clang-12-O0", "arm-clang-12-O2"),
    ("arm-gcc-8-O0", "arm-gcc-8-O2"),
    ("arm-clang-12-O0", "arm-clang-12-O3"),
    ("arm-gcc-8-O0", "arm-gcc-8-O3"),
    ("arm-clang-12-O1", "arm-clang-12-O3"),
    ("arm-gcc-8-O1", "arm-gcc-8-O3"),
    ("arm-clang-12-O2", "arm-gcc-8-O2"),
    ("arm-clang-6-O2", "arm-clang-12-O2"),
    ("arm-clang-8-O2", "arm-clang-12-O2"),
    ("arm-gcc-6-O2", "arm-gcc-10-O2"),
    ("arm-gcc-8-O2", "arm-gcc-10-O2"),
    ("x86-clang-12-O0", "arm-clang-12-O0"),
    ("x86-clang-12-O3", "arm-clang-12-O3"),
    ("x86-gcc-10-O0", "arm-gcc-10-O0"),
    ("x86-gcc-10-O3", "arm-gcc-10-O3"),
    ("x86-clang-12-O0", "arm-gcc-10-O2"),
    ("x86-gcc-8-O1", "arm-clang-6-O3"),
    ("x86-clang-12-O0", "x86-clang-12-O2"),
    ("x86-gcc-8-O0", "x86-gcc-8-O2"),
    ("x86-clang-12-O0", "x86-clang-12-O3"),
    ("x86-gcc-8-O0", "x86-gcc-8-O3"),
    ("x86-clang-12-O1", "x86-clang-12-O3"),
    ("x86-gcc-8-O1", "x86-gcc-8-O3"),
    ("x86-clang-12-O2", "x86-gcc-8-O2"),
    ("x86-clang-6-O2", "x86-clang-12-O2"),
    ("x86-clang-8-O2", "x86-clang-12-O2"),
    ("x86-gcc-6-O2", "x86-gcc-10-O2"),
    ("x86-gcc-8-O2", "x86-gcc-10-O2"),
    ("x86-clang-12-O0", "x86-gcc-10-O2"),
    ("x86-gcc-8-O1", "x86-clang-6-O3"),
    ("x86-gcc-10-O0", "x86-clang-sub-O3"),
    ("x86-gcc-10-O0", "x86-clang-bcf-O3"),
    ("x86-gcc-10-O0", "x86-clang-fla-O3"),
    ("x86-gcc-10-O0", "x86-clang-hybrid-O3"),
]

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
    parser.add_argument("-p", "--project", dest="proj", help="Project", default=None)
    parser.add_argument(
        "-o", "--out_path", dest="out_path", help="Output path", default=None
    )
    parser.add_argument(
        "-n", "--config_num", dest="config_num", help="Index to config", default=None
    )
    args = parser.parse_args()

    proj = args.proj
    base_out_path = os.path.join(
        args.out_path, proj + "-ground-truth-inline-sub-fixed-filter"
    )
    n = int(args.config_num)
    print(n)
    if n in [0, 6, 7, 8] and proj == "putty":
        test_list = test_dict_missing[proj]
    else:
        test_list = test_dict[proj]
    test_list = [bin + ".data" for bin in test_list]
    x86_data_path = args.x86_data_path
    arm_data_path = args.arm_data_path

    # configs = [('arm-clang-12-O0', 'arm-clang-12-O2'), ('arm-gcc-8-O0', 'arm-gcc-8-O2'), ('arm-clang-12-O0', 'arm-clang-12-O3'), ('arm-gcc-8-O0', 'arm-gcc-8-O3'), ('arm-clang-12-O1', 'arm-clang-12-O3'), ('arm-gcc-8-O1', 'arm-gcc-8-O3')]
    if n in range(0, 6):
        generateSet(
            os.path.join(arm_data_path, proj),
            os.path.join(arm_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp1-arm-sb"),
        )
        # configs = [('arm-clang-12-O2', 'arm-gcc-8-O2'), ('arm-clang-6-O2', 'arm-clang-12-O2'), ('arm-clang-8-O2', 'arm-clang-12-O2'), ('arm-gcc-6-O2', 'arm-gcc-10-O2'), ('arm-gcc-8-O2', 'arm-gcc-10-O2')]
    elif n in range(6, 11):
        generateSet(
            os.path.join(arm_data_path, proj),
            os.path.join(arm_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp2-arm-sb"),
        )
        # configs = [('x86-clang-12-O0', 'arm-clang-12-O0'), ('x86-clang-12-O3', 'arm-clang-12-O3'), ('x86-gcc-10-O0', 'arm-gcc-10-O0'), ('x86-gcc-10-O3', 'arm-gcc-10-O3')]
    elif n in range(11, 15):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(arm_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp3-x86-arm-sb"),
        )
        # configs = [('x86-clang-12-O0', 'arm-gcc-10-O2'), ('x86-gcc-8-O1', 'arm-clang-6-O3')]
    elif n in range(15, 17):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(arm_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp4-x86-arm-sb"),
        )
        # configs = [('x86-clang-12-O0', 'x86-clang-12-O2'), ('x86-gcc-8-O0', 'x86-gcc-8-O2'), ('x86-clang-12-O0', 'x86-clang-12-O3'), ('x86-gcc-8-O0', 'x86-gcc-8-O3'), ('x86-clang-12-O1', 'x86-clang-12-O3'), ('x86-gcc-8-O1', 'x86-gcc-8-O3')]
    elif n in range(17, 23):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(x86_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp1-x86-sb"),
        )
        # configs = [('x86-clang-12-O2', 'x86-gcc-8-O2'), ('x86-clang-6-O2', 'x86-clang-12-O2'), ('x86-clang-8-O2', 'x86-clang-12-O2'), ('x86-gcc-6-O2', 'x86-gcc-10-O2'), ('x86-gcc-8-O2', 'x86-gcc-10-O2')]
    elif n in range(23, 28):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(x86_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp2-x86-sb"),
        )
        # configs = [('x86-clang-12-O0', 'x86-gcc-10-O2'), ('x86-gcc-8-O1', 'x86-clang-6-O3')]
    elif n in range(28, 30):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(x86_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp4-x86-sb"),
        )

        # configs = [('x86-gcc-10-O0', 'x86-clang-sub-O3'), ('x86-gcc-10-O0', 'x86-clang-bcf-O3'), ('x86-gcc-10-O0', 'x86-clang-fla-O3'), ('x86-gcc-10-O0', 'x86-clang-hybrid-O3')]
    elif n in range(30, 34):
        generateSet(
            os.path.join(x86_data_path, proj),
            os.path.join(x86_data_path, proj),
            configs[n],
            test_list,
            os.path.join(base_out_path, "exp5-x86-sb"),
        )
    else:
        print("Invalid number: ", n)
