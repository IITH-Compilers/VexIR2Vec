# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Scalability Analysis Script for Normalization Time vs Total Time Plotting"""

# ex usage: python norm_time_vs_total_time.py -bin plot-filtered-40.txt -size x86-stripped-sorted-uniq-sizes.txt -ntime consolidated_norm_times_run2.txt -ttime vexir/norm-2/final-time-norm-2/final-time-x86-stripped-101-plot-filtered-real-times-fproc1-final-with-normtime.txt
# Desc: Generates plot for normalization time vs total emb gen time
# Notes: Binaries file must have paths to list of binaries considered for plot
#        Paths format <proj>/<config>/stripped/<bin>.out
#        Size file must have comma separated values for binary paths (as above), binary sizes in KB and function count of binary


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from lib2to3.pytree import convert


def prefixSum(lis):
    sum = 0.0
    for i in range(0, len(lis)):
        sum += lis[i]
        lis[i] = sum
    return lis


def getList(file):
    with open(file, "r") as f:
        time_list = [r.strip() for r in f.readlines()]
    return time_list


def convertToSec(time_list):
    sec_time_list = []
    for time in time_list:
        time = float(time.split("m")[0]) * 60 + float(time.split("m")[1][:-1])
        sec_time_list.append(time)
    return sec_time_list


def plotTimeSize(bin_file, size_file, norm_time, total_time):
    norm_time_list = getList(norm_time)
    norm_time_list = [float(element) for element in norm_time_list]
    total_time_list = getList(total_time)
    total_time_list = [float(element) for element in total_time_list]

    # Doing prefix sum
    norm_time_list = prefixSum(norm_time_list)
    total_time_list = prefixSum(total_time_list)

    # Dividing by 1000
    norm_time_list = [x / 1000 for x in norm_time_list]
    total_time_list = [x / 1000 for x in total_time_list]

    bin_list = getList(bin_file)

    with open(size_file, "r") as f:
        size_list = f.readlines()

    size_list = [tuple([val.strip() for val in r.split(",")]) for r in size_list]
    print(len(size_list))

    filtered_size_list = []
    for tup_stats in size_list:
        if tup_stats[0] in bin_list:
            filtered_size_list.append(float(tup_stats[1].strip()))

    size_arr = np.array(filtered_size_list)
    print(size_arr.shape)

    norm_time_arr = np.array(norm_time_list)
    total_time_arr = np.array(total_time_list)

    print(norm_time_arr.shape)
    print(total_time_arr.shape)

    count_arr = []
    for i in range(0, len(size_arr)):
        count_arr.append(i + 1)

    count_arr = np.array(count_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        count_arr,
        norm_time_arr,
        "-x",
        color="green",
        label="Optimization time",
        markevery=5,
        linewidth=4,
        markersize=12,
    )
    ax.plot(
        count_arr,
        total_time_arr,
        "-+",
        color="#85a3e0",
        label="VexIR2Vec #threads 1",
        markevery=5,
        linewidth=4,
        markersize=12,
    )

    fig.set_figwidth(8)
    fig.set_figheight(4)

    ax.legend(fontsize=18)

    plt.xticks([i for i in range(5, 102, 10)], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of Binaries", fontsize=16)
    plt.ylabel("Cumulative Time (Kilosec)", fontsize=16)
    plt.savefig(
        "x86-norm-total-time-n2.pdf", dpi=500, format="pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bin", "--bin_file", help="Path to file with binaries list", default=None
    )
    parser.add_argument(
        "-size",
        "--size_file",
        help="Path to file with binaries, size, function count list",
        default=None,
    )
    parser.add_argument(
        "-ntime",
        "--normalization_time",
        help="Path to file with normalization times for fproc1",
        default=None,
    )
    parser.add_argument(
        "-ttime",
        "--total_time",
        help="Path to file with total emb gen time for fproc 1",
        default=None,
    )

    args = parser.parse_args()

    plotTimeSize(
        args.bin_file, args.size_file, args.normalization_time, args.total_time
    )
