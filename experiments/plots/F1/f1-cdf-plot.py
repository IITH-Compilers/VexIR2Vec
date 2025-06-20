# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec F1 CDF Plotting Script"""

# Code for plotting f1 curves for VexIR2Vec in 2x8 format
# Eg usage: python f1-cdf-plot-new-format.py -vp /path/to/vexir-f1-score.json
# Desc: Generates F1 CDF plot as pdf for VexIR2Vec. Generates a subplot for each experiment.

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.ticker as ticker
import matplotlib.lines as mlines


def plotRoc(vexir2vec_json_path):
    if not vexir2vec_json_path:
        print("Error: Path to VexIR2Vec JSON file not provided.")
        return

    try:
        with open(vexir2vec_json_path, "r") as v:
            vexir2vec_dict = json.load(v)
    except FileNotFoundError:
        print(f"Error: VexIR2Vec JSON file not found at {vexir2vec_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode VexIR2Vec JSON file at {vexir2vec_json_path}")
        return

    exp1_arm = [
        (
            "arm-clang-12-O0-arm-clang-12-O2",
            "arm-gcc-8-O0-arm-gcc-8-O2",
            "a) ARM-O0-O2",
        ),
        (
            "arm-clang-12-O0-arm-clang-12-O3",
            "arm-gcc-8-O0-arm-gcc-8-O3",
            "b) ARM-O0-O3",
        ),
        (
            "arm-clang-12-O1-arm-clang-12-O3",
            "arm-gcc-8-O1-arm-gcc-8-O3",
            "c) ARM-O1-O3",
        ),
    ]
    exp1_x86 = [
        (
            "x86-clang-12-O0-x86-clang-12-O2",
            "x86-gcc-8-O0-x86-gcc-8-O2",
            "d) x86-O0-O2",
        ),
        (
            "x86-clang-12-O0-x86-clang-12-O3",
            "x86-gcc-8-O0-x86-gcc-8-O3",
            "e) x86-O0-O3",
        ),
        (
            "x86-clang-12-O1-x86-clang-12-O3",
            "x86-gcc-8-O1-x86-gcc-8-O3",
            "f) x86-O1-O3",
        ),
    ]
    exp2_arm = [
        ("arm-clang-12-O2-arm-gcc-8-O2", "g) ARM-Clang-GCC"),
        (
            "arm-clang-6-O2-arm-clang-12-O2",
            "arm-clang-8-O2-arm-clang-12-O2",
            "h) ARM-Clang",
        ),
        ("arm-gcc-6-O2-arm-gcc-10-O2", "arm-gcc-8-O2-arm-gcc-10-O2", "i) ARM-GCC"),
    ]
    exp2_x86 = [
        ("x86-clang-12-O2-x86-gcc-8-O2", "j) x86-Clang-GCC"),
        (
            "x86-clang-6-O2-x86-clang-12-O2",
            "x86-clang-8-O2-x86-clang-12-O2",
            "k) x86-Clang",
        ),
        ("x86-gcc-6-O2-x86-gcc-10-O2", "x86-gcc-8-O2-x86-gcc-10-O2", "l) x86-GCC"),
    ]
    exp3_x86_arm = [
        (
            "x86-clang-12-O0-arm-clang-12-O0",
            "x86-clang-12-O3-arm-clang-12-O3",
            "m) x86-ARM-Clang",
        ),
        (
            "x86-gcc-10-O0-arm-gcc-10-O0",
            "x86-gcc-10-O3-arm-gcc-10-O3",
            "n) x86-ARM-GCC",
        ),
    ]
    exp4_x86_arm = [
        (
            "x86-clang-12-O0-arm-gcc-10-O2",
            "x86-gcc-8-O1-arm-clang-6-O3",
            "o) x86-ARM-mixed",
        )
    ]
    exp4_x86 = [
        ("x86-clang-12-O0-x86-gcc-10-O2", "x86-gcc-8-O1-x86-clang-6-O3", "p) x86-mixed")
    ]
    exp_dict = {
        "exp1-arm": exp1_arm,
        "exp2-arm": exp2_arm,
        "exp1-x86": exp1_x86,
        "exp2-x86": exp2_x86,
        "exp3-x86-arm": exp3_x86_arm,
        "exp4-x86-arm": exp4_x86_arm,
        "exp4-x86": exp4_x86,
    }
    projects = ["coreutils", "diffutils", "findutils", "curl", "lua", "putty", "gzip"]
    f1_axis = [i / 20000 for i in range(0, 20001, 1)]
    f1_axis = np.array(f1_axis)
    fig, axs = plt.subplots(2, 8, sharex="all", sharey="all", figsize=(7, 16))
    plot_count = 0

    found_any_vexir2vec_data = False

    for exp in [
        "exp1-arm",
        "exp1-x86",
        "exp2-arm",
        "exp2-x86",
        "exp3-x86-arm",
        "exp4-x86-arm",
        "exp4-x86",
    ]:
        for config_tuple in exp_dict[exp]:
            vexir2vec_f1_scores = []

            config_list = []
            config_title = config_tuple[-1]
            config_count = 0
            while config_count < len(config_tuple) - 1:
                config_list.append(config_tuple[config_count])
                config_count += 1

            for config in config_list:
                for proj in projects:
                    if (
                        exp in vexir2vec_dict
                        and proj in vexir2vec_dict[exp]
                        and config in vexir2vec_dict[exp][proj]
                        and isinstance(vexir2vec_dict[exp][proj][config], list)
                    ):
                        vexir2vec_f1_scores.extend(vexir2vec_dict[exp][proj][config])

            print(config_title)
            print("VexIR2Vec F1 scores collected: ", len(vexir2vec_f1_scores))

            vexir2vec_f1_scores = np.sort(np.array(vexir2vec_f1_scores))
            if len(vexir2vec_f1_scores) > 0:
                found_any_vexir2vec_data = True

            vexir2vec_cdf = []

            if len(vexir2vec_f1_scores) > 0:
                vexir2vec_cdf = np.searchsorted(vexir2vec_f1_scores, f1_axis) / len(
                    vexir2vec_f1_scores
                )

            row_index = plot_count % 2
            col_index = plot_count // 2

            if len(vexir2vec_f1_scores) > 0:
                axs[row_index][col_index].plot(
                    f1_axis,
                    vexir2vec_cdf,
                    linewidth=0.8,
                    linestyle="-",
                    color="blue",
                    label="VexIR2Vec",
                )

            axs[row_index][col_index].set_title(config_title, fontsize=4, pad=1)
            axs[row_index][col_index].tick_params(
                width=0.2, grid_alpha=0.5, labelsize=4
            )
            axs[row_index][col_index].set_xticks([0, 0.5, 1], minor=False)
            axs[row_index][col_index].set_yticks([0, 0.5, 1], minor=False)
            axs[row_index][col_index].xaxis.set_major_formatter(
                ticker.FormatStrFormatter("%g")
            )
            axs[row_index][col_index].yaxis.set_major_formatter(
                ticker.FormatStrFormatter("%g")
            )
            axs[row_index][col_index].set_xlim([-0.08, 1.08])
            axs[row_index][col_index].set_ylim([-0.08, 1.08])
            axs[row_index][col_index].grid(linestyle="--", which="both", linewidth=0.5)
            axs[row_index][col_index].spines["top"].set_linewidth(0.5)
            axs[row_index][col_index].spines["right"].set_linewidth(0.5)
            axs[row_index][col_index].spines["bottom"].set_linewidth(0.5)
            axs[row_index][col_index].spines["left"].set_linewidth(0.5)

            plot_count += 1

    handles = []
    labels = []
    if found_any_vexir2vec_data:
        handles.append(
            mlines.Line2D([], [], color="blue", linestyle="-", label="VexIR2Vec")
        )
        labels.append("VexIR2Vec")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.05),
            fontsize=5,
        )
    else:
        # This message is shown if the JSON was valid but no data matched experiments
        print(
            "No VexIR2Vec data found for the specified experiment configurations. Plot may be empty."
        )

    fig.text(0.50, -0.02, "F1 score", ha="center", fontsize=6)
    fig.text(0.08, 0.45, "Fraction data", va="center", rotation=90, fontsize=6)
    fig.set_figwidth(7)
    fig.set_figheight(2)
    text_x = -4.33
    text_y = -0.76
    text_content = "a-f:  Cross-Optimization; g-l: Cross-Compiler; m-n: Cross-Architecture; o-p: Cross-Compiler + Cross-Optimization + Cross-Architecture"

    plt.text(
        text_x,
        text_y,
        text_content,
        fontsize=6,
        color="black",
        ha="center",
        va="center",
    )
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    plt.savefig("F1-cdf-plot-vexir2vec.pdf", dpi=300, format="pdf", bbox_inches="tight")
    # plt.show() # Uncomment for interactive display if needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot F1 CDF curves for VexIR2Vec.")
    parser.add_argument(
        "-vp",
        "--vexir2vec_json",
        help="Path to VexIR2Vec F1-score json file",
        default=None,
    )
    args = parser.parse_args()

    plotRoc(args.vexir2vec_json)
