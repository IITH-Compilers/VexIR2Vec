# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Scalability Analysis Script for Time vs Size Plotting"""

# Usage: python3 time_vs_size_vexir2vec.py -bin <Path to file with binaries list> -size <Path to file with binaries, size, function counts> -vp1 <Path to VexIR2Vec times for #threads=1> -vp2 <Path to VexIR2Vec times for #threads=2> -vp4 <Path to VexIR2Vec times for #threads=4> -vp8 <Path to VexIR2Vec times for #threads=8>
# Desc: Generates plot for embedding generation time vs binary size for VexIR2Vec
# Notes: Binaries file must have paths to binaries considered for plot
#        Size file must contain comma-separated values: binary path, size in KB, function count
#        VexIR2Vec time files must contain real time in seconds per binary (one per line)

import argparse
import numpy as np
import matplotlib.pyplot as plt


def prefixSum(lis):
    total = 0.0
    for i in range(len(lis)):
        total += lis[i]
        lis[i] = total
    return lis


def getList(file):
    with open(file, "r") as f:
        return [r.strip() for r in f.readlines()]


def plotTimeSize(
    bin_file,
    size_file,
    vexir2vec_time1,
    vexir2vec_time2,
    vexir2vec_time4,
    vexir2vec_time8,
):
    vex_time_list1 = [float(x) for x in getList(vexir2vec_time1)]
    vex_time_list2 = [float(x) for x in getList(vexir2vec_time2)]
    vex_time_list4 = [float(x) for x in getList(vexir2vec_time4)]
    vex_time_list8 = [float(x) for x in getList(vexir2vec_time8)]

    vex_time_list1 = [x / 1000 for x in prefixSum(vex_time_list1)]
    vex_time_list2 = [x / 1000 for x in prefixSum(vex_time_list2)]
    vex_time_list4 = [x / 1000 for x in prefixSum(vex_time_list4)]
    vex_time_list8 = [x / 1000 for x in prefixSum(vex_time_list8)]

    bin_list = getList(bin_file)

    with open(size_file, "r") as f:
        size_list = f.readlines()

    size_list = [tuple(r.strip().split(",")) for r in size_list]
    filtered_size_list = [float(tup[1]) for tup in size_list if tup[0] in bin_list]

    count_arr = np.arange(1, len(filtered_size_list) + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        count_arr,
        vex_time_list1,
        "-+",
        color="#85a3e0",
        label="VexIR2Vec #threads 1",
        markevery=5,
    )
    ax.plot(
        count_arr,
        vex_time_list2,
        "-*",
        color="#00ace6",
        label="VexIR2Vec #threads 2",
        markevery=5,
    )
    ax.plot(
        count_arr,
        vex_time_list4,
        "-d",
        color="#8000ff",
        label="VexIR2Vec #threads 4",
        markevery=5,
    )
    ax.plot(
        count_arr,
        vex_time_list8,
        "-o",
        color="blue",
        label="VexIR2Vec #threads 8",
        markevery=5,
    )

    fig.set_figwidth(10)
    fig.set_figheight(4)

    ax.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of Binaries", fontsize=16)
    plt.ylabel("Cumulative Time (Kilosec)", fontsize=16)
    plt.savefig(
        "x86-time-size-cumulative-vexir2vec.pdf",
        dpi=500,
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bin", "--bin_file", help="Path to file with binaries list", required=True
    )
    parser.add_argument(
        "-size",
        "--size_file",
        help="Path to file with binaries, size, function count list",
        required=True,
    )
    parser.add_argument(
        "-vp1",
        "--vexir2vec_time1",
        help="Path to file with VexIR2Vec times for #threads=1",
        required=True,
    )
    parser.add_argument(
        "-vp2",
        "--vexir2vec_time2",
        help="Path to file with VexIR2Vec times for #threads=2",
        required=True,
    )
    parser.add_argument(
        "-vp4",
        "--vexir2vec_time4",
        help="Path to file with VexIR2Vec times for #threads=4",
        required=True,
    )
    parser.add_argument(
        "-vp8",
        "--vexir2vec_time8",
        help="Path to file with VexIR2Vec times for #threads=8",
        required=True,
    )

    args = parser.parse_args()

    plotTimeSize(
        args.bin_file,
        args.size_file,
        args.vexir2vec_time1,
        args.vexir2vec_time2,
        args.vexir2vec_time4,
        args.vexir2vec_time8,
    )
