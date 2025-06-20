# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Search Wrapper Script for Searching Experiment"""

# Usage: python3 v2v_search_wrapper.py -bmp /path/to/.model -dp /path/to/x86-data-all -search_gt_dir /path/to/GroundTruth -res_dir /path/to/store_results -out_dir /path/to/store_roc -n <threads> -chunks <num_chunks>

import argparse
import subprocess
import sys
import os
from v2v_searching import generateSearchSpace, getSearchScores

sys.path.append(os.path.abspath("../../embeddings/vexNet"))
from utils_inference import printVex2vec


# Default values for projects and configs
test_list = ["findutils", "diffutils", "coreutils", "gzip", "lua", "curl", "putty"]
gt_configs = ["x86-clang-8", "x86-clang-12", "x86-gcc-8", "x86-gcc-10"]


def main():
    parser = argparse.ArgumentParser(description="Wrapper script for V2V.py")

    parser.add_argument(
        "-bmp",
        "--best_model_path",
        default="/path/to/sample.model",
        help="Path to the best model",
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        default="/path/to/x86-data-all",
        help="Directory containing data files of all the projects.",
    )
    parser.add_argument(
        "-search_gt_dir",
        "--binsearch_test_dir",
        default="/path/to/Groundtruth",
        help="Path to ground truth csv",
    )
    parser.add_argument(
        "-res_dir",
        "--res_dir",
        default="/path/to/results",
        help="Path to output directory to store results",
    )
    parser.add_argument(
        "-out_dir",
        "--out_dir",
        default="/path/to/output",
        help="Path to output directory of roc json files",
    )
    parser.add_argument(
        "-cfg",
        "--use_cfg",
        type=bool,
        default=False,
        help="Use CFG data for inferencing if set",
    )
    parser.add_argument(
        "-n", "--threads", type=int, default=20, help="Number of threads to use"
    )
    parser.add_argument(
        "-chunks",
        "--num_chunks",
        type=int,
        default=100,
        help="Number of chunks to split the data into",
    )

    parser.add_argument(
        "-filter",
        help="Specify package names separated by spaces or use 'all-packages' to include all.",
        default="all-packages",
    )
    parser.add_argument(
        "-config",
        help="Specify configs names separated by spaces or use 'all-configs' to include all.",
        default="all-configs",
    )

    args = parser.parse_args()

    printVex2vec()  # Print VexIR2Vec info

    # Generate search space with all arguments
    search_key_embed_dict = generateSearchSpace(args)
    # print(search_key_embed_dict)
    # exit(0)

    if args.filter == "all-packages":
        packages = test_list  # Use predefined test list
    else:
        packages = args.filter.split(",")

    if args.config == "all-configs":
        configs = gt_configs
    else:
        configs = args.config.split(",")

    for package in packages:
        if package not in test_list:
            continue

        print(f"Processing package: {package }")

        getSearchScores(
            args,
            args.binsearch_test_dir,
            args.num_chunks,
            search_key_embed_dict,
            package,
            configs,
        )


if __name__ == "__main__":
    main()
