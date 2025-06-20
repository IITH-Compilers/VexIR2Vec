# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec OOV Plotting Script for OOV Counts"""

# Usage: python3 oov-plot.py -vp <Path to VexIR2Vec OOV txt file>
# Desc: Generates OOV plot for VexIR2Vec for all x86-clang-12-O0 binaries in binutils, coreutils, diffutils, findutils and openssl

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plotOov(vex_file):
    file_name = [i for i in range(1, 398)]

    with open(vex_file, "r") as f:
        vex_lines = [int(r.strip()) for r in f.readlines()]

    # Sort based on VexIR2Vec OOV counts
    sorted_vex_list = sorted(vex_lines)
    vex_oov = np.array(sorted_vex_list)
    file_name = np.array(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale("symlog")
    ax.margins(y=0.1)

    ax.plot(file_name, vex_oov, "-o", color="blue", label="VexIR2Vec", markevery=20)

    fig.set_figwidth(8)
    fig.set_figheight(4)

    plt.rc("legend", fontsize=14)
    plt.legend(loc="upper left")
    plt.yticks(fontsize=11)
    plt.xlabel("Binaries", fontsize=14)
    plt.ylabel("OOV count", fontsize=14)
    plt.savefig("vexir2vec-oov.pdf", dpi=500, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vp",
        "--vexir2vec_file",
        help="Path to VexIR2Vec unknown entity count",
        default=None,
    )
    args = parser.parse_args()

    plotOov(args.vexir2vec_file)
