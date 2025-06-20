#!/bin/bash

# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# VexIR2Vec Diffing Wrapper Script

# Usage: ./v2v_wrapper_diffing.sh GROUND_TRUTH_PATH GT_SUFFIX INFERENCE_PATH MODEL_PATH DATA_FILES_PATH NUM_THREADS PROJECTS
# Configuration
GROUND_TRUTH_PATH=$1
GT_SUFFIX=$2
INFERENCE_PATH=$3
MODEL=$4
DP=$5
NUM_THREADS=$6
projects=($7)

for PROJECT in "${projects[@]}"; do
	echo "[INFO] Processing project: $PROJECT"

	proj=${PROJECT%%-*}
	gt_path="${GROUND_TRUTH_PATH}/${PROJECT}${GT_SUFFIX}"
	data_path="${DP}/${PROJECT}/"
	out_dir="${INFERENCE_PATH}/roc"
	res_dir="${INFERENCE_PATH}/results"

	csv_paths=()

	# Collect CSVs for experiments 1-2: arm-sb and x86-sb
	for n in $(seq 1 2); do
		for csv_path in ${gt_path}/exp$n-arm-sb/*.csv; do
			csv_paths+=("$csv_path")
		done
		for csv_path in ${gt_path}/exp$n-x86-sb/*.csv; do
			csv_paths+=("$csv_path")
		done
	done

	# Collect CSVs for experiments 3-4: x86-arm-sb
	for n in $(seq 3 4); do
		for csv_path in ${gt_path}/exp$n-x86-arm-sb/*.csv; do
			csv_paths+=("$csv_path")
		done
	done

	# Experiment 4: x86-sb
	for csv_path in ${gt_path}/exp4-x86-sb/*.csv; do
		csv_paths+=("$csv_path")
	done

	# Experiment 5: obfuscated x86-sb
	for csv_path in ${gt_path}/exp5-x86-sb/*.csv; do
		csv_paths+=("$csv_path")
	done

	echo "[INFO] Found ${#csv_paths[@]} CSVs for $PROJECT"

	# Run inference in parallel
	parallel -j $NUM_THREADS \
		python v2v_diffing.py \
		-bmp $MODEL \
		-dp $data_path \
		-test_csv {} \
		-out_dir $out_dir \
		-res_dir $res_dir ::: "${csv_paths[@]}"
done
