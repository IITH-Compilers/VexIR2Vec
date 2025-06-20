#!/bin/bash

# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# This is the Wrapper script generating the database files

# Usage:
# bash genMetaDataDB.sh <path-to-binaries> <num-threads> <output-db-path>

start=$(date)
echo "----------------------------------------------------------------------------------------------------------------------"

echo "Script start time: $start"
echo "-"

PROJ_BINS_PATH=$1
NUM_THREADS=$2
DB_PATH=$3
echo "PROJ_BINS_PATH: $PROJ_BINS_PATH"
if [ -z "$PROJ_BINS_PATH" ]; then
	echo "set SOURCE_PATH as 1st argument"
	exit
fi

function extractBinData() {
	bin_path=$1
	THREADS=$3
	project=$2
	type=$4
	DB_PATH=$5
	echo $THREADS
	echo $DB_PATH

	data_dir="$(dirname $bin_path)/$4-$2-db"

	mkdir -p $data_dir
	echo $data_dir
	echo "bin_dir_path: $bin_path"

	python angr_db_metadata_dump.py -b_dir "$bin_path" -adb "$data_dir" -edb "$DB_PATH" -t "$THREADS"

}
export -f extractBinData

if [[ "$PROJ_BINS_PATH" == *"arm"* ]]; then
	archs=('arm')
fi

if [[ "$PROJ_BINS_PATH" == *"x86"* ]]; then
	archs=('x86')
fi

projects=('diffutils') # Your Project name
compilers=('clang-12')
opts=('O0')

for proj in ${projects[@]}; do
	for arch in ${archs[@]}; do
		for compiler in ${compilers[@]}; do
			for opt in ${opts[@]}; do
				conf=$arch-$compiler-$opt
				dest="$PROJ_BINS_PATH"/"$proj"/"$conf"
				echo "Conf: $conf"
				echo "Dest: $dest"
				if [ -d $(realpath ${dest}/unstripped) ]; then
					extractBinData $(realpath ${dest}/unstripped) $proj $NUM_THREADS unstripped $DB_PATH
				fi
				wait
				if [ -d $(realpath ${dest}/stripped) ]; then

					extractBinData $(realpath ${dest}/stripped) $proj $NUM_THREADS stripped $DB_PATH
				fi
				wait
			done
		done
	done
done

wait

echo "-"
end=$(date)

echo "Script end time: $end"
echo "-----------------------------------------------------------------------------------------------------------------"
