# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# VexIR2Vec Test Set Generation Wrapper Script for Diffing Experiment

# Usage: bash testset-gen-wrapper.sh <Project-name> <Path to x86 data> <Path to ARM data> <Output path> $NUM_THREADS
# Ex: bash testset-gen-wrapper.sh findutils /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-diffing-groundtruth/ 30
# Desc: Generates a directory for the project, containing ground truth csv for each experiment config, arranged experiment wise
# NOTE: seq passed config number as input to GNU parallel

PROJ=$1
x86_DATA=$2
ARM_DATA=$3
OUTPUT_DIR=$4
NUM_THREADS=$5

seq 0 33 | parallel -j$NUM_THREADS python3 test-set-gen-nolib.py -xd $x86_DATA -ad $ARM_DATA -p $PROJ -o $OUTPUT_DIR -n {}
