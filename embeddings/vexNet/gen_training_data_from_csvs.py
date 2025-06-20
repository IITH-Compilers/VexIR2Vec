# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Used to generate the training data h5 file"""

# Usage:
# python gen_training_data_from_csvs.py <space separated projects>
# ex: python gen_training_data_from_csvs.py findutils diffutils

import os
import sys
import h5py
import numpy as np
import pandas as pd
import argparse
from sklearn.utils import shuffle
from utils import *
import ast
from datetime import datetime
from collections import defaultdict
from IncrementalHDF5 import IncrementalHDF5


def genH5(df):

    print("Generating h5 file...")
    hf = h5py.File("/path/to/output/h5/file", "w")
    hf.create_dataset("keys", data=df.key.values.astype("S"))
    hf.create_dataset("strRefs", data=np.stack(df.strRefs.values))
    hf.create_dataset("extlibs", data=np.stack(df.extlibs.values))
    hf.create_dataset("opc_embed", data=np.stack(df.embed_O.values))
    hf.create_dataset("type_embed", data=np.stack(df.embed_T.values))
    hf.create_dataset("arg_embed", data=np.stack(df.embed_A.values))
    hf.close()


def preProcess(data, wo=1, wt=1, wa=1):

    data["key"] = data["key"].astype("S")
    data.drop(["addr"], axis=1, inplace=True)
    data.drop_duplicates(inplace=True, ignore_index=True)

    with h5py.File(KEYS_FILE, "r") as hf:

        keys_dataset = hf["keys"]
        key_counts = defaultdict(int)

        # Iterate over the values in the dataset and count their occurrences
        for value in keys_dataset:
            # Decode the value if it's in bytes format
            if isinstance(value, bytes):
                value = value.decode()
                # Increment the count for the value
            key_counts[value] += 1

            # Convert the defaultdict to a regular dictionary
    key_counts_dict = dict(key_counts)
    # Get the keys from key_counts_dict
    valid_keys = set(key_counts_dict.keys())

    print("dataframe size before key filtering: ", len(data))

    valid_keys_bytes = {x.encode() for x in valid_keys}

    data = data[data["key"].isin(valid_keys_bytes)]

    keys = data.key.value_counts()
    idxs = keys[
        keys >= 30
    ].index  # to ensure only keys with occurrences > =30 are remaining in dataframe
    data = data[data["key"].isin(idxs)]

    print("dataframe size after key filtering: ", len(data))

    data.drop(["strRefs", "extlibs", "embedding"], axis=1, inplace=True)
    data.cfg = data.cfg.astype(str)

    for i, value in enumerate(data.cfg):
        try:
            data["cfg"][i] = strToMuldimNpArr(value)
        except Exception as e:
            print(f"Error processing row {i }: {e }")
            print(f"Value of column key in row {i }: {data .loc [i ,'key']}")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-inc-wr",
        "--incremental-write",
        dest="incremental_write",
        required=False,
        default=False,
        action="store_true",
        help="Write to H5 file incrementally",
    )
    parser.add_argument(
        "-findutils",
        "--findutils",
        dest="findutils",
        required=False,
        default=False,
        action="store_true",
        help="Include this project",
    )
    parser.add_argument(
        "-diffutils",
        "--diffutils",
        dest="diffutils",
        required=False,
        default=False,
        action="store_true",
        help="Include this project",
    )
    parser.add_argument(
        "-coreutils",
        "--coreutils",
        dest="coreutils",
        required=False,
        default=False,
        action="store_true",
        help="Include this project",
    )

    parser.add_argument(
        "-wo",
        "--wo",
        dest="wo",
        default=1.0,
        type=float,
        required=False,
        help="Opcode Weights",
    )
    parser.add_argument(
        "-wt",
        "--wt",
        dest="wt",
        default=1.0,
        type=float,
        required=False,
        help="Type Weights",
    )
    parser.add_argument(
        "-wa",
        "--wa",
        dest="wa",
        default=1.0,
        type=float,
        required=False,
        help="Argument Weights",
    )

    args = parser.parse_args()

    projects, dataset = [], []

    if args.findutils:
        projects.append("findutils")
    if args.diffutils:
        projects.append("diffutils")
    if args.coreutils:
        projects.append("coreutils")

    print(f"Projects: {projects }")
    test_dict = collect_test_bins()

    incremental_hdf5 = None

    for proj in projects:
        if args.incremental_write:
            h5file = os.path.join(TRAIN_DATA_H5FILE_DIR, proj + ".h5")
            incremental_hdf5 = IncrementalHDF5(
                h5file,
                target_size=1024 * 1024,
                preprocess_fn=preProcess,
                wo=args.wo,
                wt=args.wt,
                wa=args.wa,
            )
            # Collecting data from data files
        print(f"Generating training data h5 for {proj } ...")
        x86_proj_path = os.path.join(X86_DATA_PATH, proj)
        arm32_proj_path = os.path.join(ARM_DATA_PATH, proj)

        x86_data = loadData(
            x86_proj_path,
            test_dict=test_dict,
            use_cfg=args.use_cfg,
            hf=incremental_hdf5,
        )
        print("x86 data loaded.")

        arm32_data = loadData(
            arm32_proj_path,
            test_dict=test_dict,
            use_cfg=args.use_cfg,
            hf=incremental_hdf5,
        )
        print("arm32 data loaded.")

        if args.incremental_write:
            incremental_hdf5.close()
            exit(1)

        else:
            # Merging the extracted data
            data = pd.concat([x86_data, arm32_data], axis=0, ignore_index=True)
            data = preProcess(data, args.wo, args.wt, args.wa)
            print(f"For {proj }, data shape: {data .shape }")
            dataset.append(data)

            # Creating H5 file for the training set
    dataset = pd.concat(dataset)

    genH5(dataset)
