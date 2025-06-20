#!/bin/bash
# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# This is the wrapper script that is used to create the triplets
# cmd: bash genAngrTriplets.sh <num of THREADS to be used> <Dir containing all the binaries> <Dest-dir>
# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
	echo "Usage: $0 <num_threads> <bin_directory> <dest_directory>"
	exit 1
fi

num_threads="$1"
BIN_DIR=$(realpath "$2")
DST_DIR=$(realpath "$3")
host=$(hostname)
TRIPLET_DIR="${DST_DIR}/${host%%.*}_angr_triplets"

echo "Number of threads to be used: ${num_threads}"
echo "Src directory: ${BIN_DIR}"
echo "Dest directory: ${DST_DIR}"
echo "Deduplicated Triplets are saved at: ${TRIPLET_DIR}"

# Check if source directory exists
if [ ! -d "$BIN_DIR" ]; then
	echo "Error: Source directory '$BIN_DIR' does not exist"
	exit 1
fi

# Create destination directory
mkdir -p "$TRIPLET_DIR"

# Export variables so they're available in parallel subprocesses
export TRIPLET_DIR
export BIN_DIR

# Define the command
cmd="[[ ! -f \$TRIPLET_DIR/{} ]] && python ../../embeddings/pre-training/driver.py -b \$BIN_DIR/{} -t -o \$TRIPLET_DIR/{} -mode non-db"

# Run parallel processing
ls "$BIN_DIR" | parallel -j"${num_threads}" "${cmd}"
wait
