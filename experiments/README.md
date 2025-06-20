# Experiments

This directory contains all scripts for performing the experiments described in the VexIR2Vec paper.
It includes binary similarity tasks such as **diffing** and **searching**, scalability analysis, and OOV study.

If you want to study the diffing and searching results, you may need to generate the ground truth. Ground truth
is characterized by CSV files. Only functions named `sub_<addr>` or `main` are considered.  Functions lacking
source metadata (e.g., `NO_FILE_SRC`) are excluded. For details, refer to the
[groundtruth](./groundtruth) directory.
