# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

""" Creates a single combined H5 file for mentioned projects containing keys(string)"""

# Usage:
# python gen_keys_h5.py <projects separated by spaces>
# ex: python gen_keys_h5.py findutils diffutils


import os
import sys
import h5py
import pandas as pd
from utils import (
    collect_test_bins,
    STRIPPED_DIR,
    UNSTRIPPED_DIR,
    KEYS_FILE,
    X86_DATA_PATH,
    ARM_DATA_PATH,
)


def loadFiles(unst_path, st_path, test_dict):
    file_dfs = []
    for unst_file, st_file in zip(
        sorted(os.listdir(unst_path)), sorted(os.listdir(st_path))
    ):
        proj = os.path.basename(os.path.dirname(os.path.dirname(unst_path)))
        if unst_file not in test_dict[proj]:
            col_names = ["addr", "key", "fnName"]
            unst_df = pd.read_csv(
                os.path.join(unst_path, unst_file),
                sep="\t",
                usecols=[0, 1, 2],
                names=col_names,
                header=None,
            )
            unst_df = unst_df[unst_df["key"].str.contains("NO_FILE_SRC") == False]
            unst_df.reset_index(drop=True, inplace=True)
            unst_df.addr.astype(str)

            st_df = pd.read_csv(
                os.path.join(st_path, st_file),
                sep="\t",
                usecols=[0, 1, 2],
                names=col_names,
                header=None,
            )
            st_df.addr.astype(str)

            file_df = pd.merge(unst_df, st_df, on="addr", suffixes=("_unst", "_st"))
            file_df.key_unst = file_df.key_unst + file_df.fnName_unst
            file_df.drop(
                ["addr", "fnName_unst", "key_st", "fnName_st"], axis=1, inplace=True
            )
            file_dfs.append(file_df)

    return file_dfs


def loadDatasets(proj_path, test_dict):
    unst_path_list, st_path_list, dfs = [], [], []
    if os.path.isdir(proj_path):
        for config in os.listdir(proj_path):
            data_path = os.path.join(proj_path, config)
            unst_path_list.append(os.path.join(data_path, UNSTRIPPED_DIR))
            st_path_list.append(os.path.join(data_path, STRIPPED_DIR))

    for unst_path, st_path in zip(unst_path_list, st_path_list):
        try:
            dataset_df = pd.concat(loadFiles(unst_path, st_path, test_dict))
            dfs.append(dataset_df)

        except Exception as err:
            print(err)

    dfs = pd.concat(dfs)
    return dfs


n = len(sys.argv)
projects = []
for i in range(1, n):
    projects.append(sys.argv[i])
print(f"Projects: {projects }")

print(f"Generating training data key H5 file")
test_dict = collect_test_bins()
keys = []
for proj in projects:
    x86_proj_path = os.path.join(X86_DATA_PATH, proj)
    arm32_proj_path = os.path.join(ARM_DATA_PATH, proj)
    x86_data = loadDatasets(x86_proj_path, test_dict=test_dict)
    print("x86 data loaded.")
    arm32_data = loadDatasets(arm32_proj_path, test_dict=test_dict)
    print("arm32 data loaded.")
    data = pd.concat([x86_data, arm32_data], axis=0, ignore_index=True)
    keys.append(data)
    print(f"{proj } loaded.")

keys = pd.concat(keys)

key_cnt = keys.key_unst.value_counts()

idxs = key_cnt[
    key_cnt >= 30
].index  # to ensure only keys with occurrences > =30 are remaining in dataframe
keys = keys[keys["key_unst"].isin(idxs)]

new_key_cnt = keys.key_unst.value_counts()

keys.drop_duplicates(inplace=True, ignore_index=True)


print(f"Total number of keys: {keys .shape }")
print(f"Sample: {keys .head (100 )}")
hf = h5py.File(KEYS_FILE, "w")
hf.create_dataset("keys", data=keys.key_unst.values.astype("S"))
hf.close()
