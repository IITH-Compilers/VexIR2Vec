#!/bin/bash
# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# Main wrapper script to genrate the embeddings

# Eg. usage: bash genDataInArray.sh /path/to/x86-data-all/ 5 /path/to/seed-embedding/ /path/to/embedding/project.db 2 2 db
start=$(date)
echo "----------------------------------------------------------------------------------------------------------------------"
echo "Script start time: $start"
echo "-"

PROJ_BINS_PATH=$1
NUM_THREADS=$2
SEED_VOCAB_PATH=$3
if [ "$6" == "non-db" ]; then
	MODE="non-db"
	NORM=$4
	NUM_FUNC_CHUNKS=$5
else
	EMBEDDING_DB_PATH=$4
	NORM=$5
	NUM_FUNC_CHUNKS=$6
	MODE=$7
fi

echo "PROJ_BINS_PATH: $PROJ_BINS_PATH"
if [ -z "$PROJ_BINS_PATH" ]; then
	echo "set SOURCE_PATH as 1st argument"
	exit
fi

function extractBinData() {
	bin_path=$1
	SEED_VOCAB_PATH=$2
	NORM=$3
	NUM_FUNC_CHUNKS=$4
	EMBEDDING_DB_PATH=$5
	MODE=$6
	echo $EMBEDDING_DB_PATH
	# echo $MODE
	# exit 0

	data_dir="$(dirname $bin_path)-suffix"
	mkdir -p $data_dir
	bin_name=$(basename $bin_path)
	if [[ ! $bin_name =~ "test" ]]; then
		echo "bin_name: $bin_name"
		check_data_file="$data_dir/${bin_name%.out*}.data"
		check_data_file_size=$(wc -c <$check_data_file)
		if [ -f "$check_data_file" ] && [ "$check_data_file_size" -ne 0 ]; then
			echo "${bin_name%.out*}.data exists"

		else
			echo "$check_data_file is empty, we are proceeding"
			if [ "$MODE" == "db" ]; then
				python ../driver.py -b $bin_path -o $data_dir/${bin_name%.out*}.data -v ${SEED_VOCAB_PATH} -d -n ${NORM} -fchunks ${NUM_FUNC_CHUNKS} -edb $EMBEDDING_DB_PATH -mode $MODE
			else
				python ../driver.py -b $bin_path -o $data_dir/${bin_name%.out*}.data -v ${SEED_VOCAB_PATH} -d -n ${NORM} -fchunks ${NUM_FUNC_CHUNKS} -mode non-db
			fi
		fi
	else
		echo "Skipping driver.py for binary containing 'test': $bin_name"
	fi
}
export -f extractBinData

if [[ "$PROJ_BINS_PATH" == *"arm"* ]]; then
	archs=('arm')
fi

if [[ "$PROJ_BINS_PATH" == *"x86"* ]]; then
	archs=('x86')
fi

echo "Running"

projects=('diffutils') # MENTION YOUR PROJECT
compilers=('clang-12') # MENTION YOUR COMPILER
opts=('O0')            # MENTION THE OPTIMIZATION

for proj in ${projects[@]}; do
	for arch in ${archs[@]}; do
		for compiler in ${compilers[@]}; do
			for opt in ${opts[@]}; do
				conf=$arch-$compiler-$opt
				dest="$PROJ_BINS_PATH"/"$proj"/"$conf"
				echo "Conf: $conf"
				echo "Dest: $dest"
				realpath ${dest}/unstripped/* | parallel -j $NUM_THREADS extractBinData {} $SEED_VOCAB_PATH $NORM $NUM_FUNC_CHUNKS $EMBEDDING_DB_PATH $MODE
				wait
				realpath ${dest}/stripped/* | parallel -j $NUM_THREADS extractBinData {} $SEED_VOCAB_PATH $NORM $NUM_FUNC_CHUNKS $EMBEDDING_DB_PATH $MODE
				wait
			done
		done
	done
done

wait

#conda deactivate
echo "-"
end=$(date)
# echo "Slurm Script end time: $end"
echo "Script end time: $end"
echo "-----------------------------------------------------------------------------------------------------------------"
