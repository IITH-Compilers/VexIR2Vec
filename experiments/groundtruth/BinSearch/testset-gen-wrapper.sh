#!/bin/bash

# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# VexIR2Vec Test Set Generation Wrapper Script for Binary Search Experiment

# Usage: bash testset-gen-wrapper.sh <Path to x86 data> <Path to ARM data> <Output path> $NUM_THREADS
# Ex: bash testset-gen-wrapper.sh /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/searching-groundtruth/ 30
# Desc: Generates a directory for the project, containing ground truth csv for each experiment config, arranged experiment wise
# NOTE: seq passed config number as input to GNU parallel

x86_DATA=$1
ARM_DATA=$2
OUTPUT_DIR=$3
NUM_THREADS=$4

seq 0 23 | parallel -j$NUM_THREADS python3 test-set-gen-binsearch.py -xd $x86_DATA -ad $ARM_DATA -o $OUTPUT_DIR -n {}
