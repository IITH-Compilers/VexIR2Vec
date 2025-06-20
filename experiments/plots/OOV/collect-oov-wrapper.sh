# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# VexIR2Vec OOV Count Collection Wrapper Script

# Usage: bash collect-oov-wrapper.sh <Path to binaries directory> <Path to VexIR directory> <VexIR OOV output file> <NUM_THREADS>
# ex: bash collect-oov-wrapper.sh /path/to/x86-binaries-all /path/to/vexIR vexir2vec-oov.txt 20
# Desc: Runs VexIR2Vec function vector generation and counts number of unknown entities (OOV) for each binary
# Notes:
#        Modified scripts: vexir2vec-collect-oov-count.py, embeddings_collect_oov_count.py, entities_collect_oov_count.py
#        Check Seed Embedding path

base_bins_path=$1
VEXIR_DIR=$2
VEXIR_OUTPUT_FILE=$3
NUM_THREADS=$4
VEXIR2VEC_PATH=$VEXIR_DIR/angr-vex
VOCAB_PATH=$VEXIR_DIR/path/to/seed_embeddings # Set the correct path

# Binary lists

coreutils_test_list=('b2sum' 'base32' 'base64' 'basename' 'basenc' 'bench-md5' 'bench-sha1' 'bench-sha224' 'bench-sha256' 'bench-sha384' 'bench-sha512' 'cat' 'chcon' 'chgrp' 'chmod' 'chown' 'chroot' 'cksum' 'comm' 'cp' 'csplit' 'current-locale' 'cut' 'date' 'dd' 'df' 'dircolors' 'dirname' 'dir' 'du' 'echo' 'env' 'expand' 'expr' 'factor' 'false' 'fmt' 'fold' 'getlimits' 'ginstall' 'groups' 'head' 'hostid' 'id' 'join' 'kill' 'link' 'ln' 'logname' 'ls' 'make-prime-list' 'md5sum' 'mkdir' 'mkfifo' 'mknod' 'mktemp' 'mv' 'nice' 'nl' 'nohup' 'nproc' 'numfmt' 'od' '[' 'paste' 'pathchk' 'pinky' 'printenv' 'printf' 'pr' 'ptx' 'pwd' 'readlink' 'realpath' 'rmdir' 'rm' 'runcon' 'seq' 'sha1sum' 'sha224sum' 'sha256sum' 'sha384sum' 'sha512sum' 'shred' 'shuf' 'sleep' 'sort' 'split' 'stat' 'stdbuf' 'stty' 'sum' 'sync' 'tac' 'tail' 'tee' 'test-localcharset' 'test' 'timeout' 'touch' 'tr' 'true' 'truncate' 'tsort' 'tty' 'uname' 'unexpand' 'uniq' 'unlink' 'uptime' 'users' 'vdir' 'wc' 'whoami' 'who' 'yes')

diffutils_test_list=('cmp' 'current-locale' 'diff3' 'diff' 'sdiff' 'test-localcharset')
findutils_test_list=('current-locale' 'find' 'frcode' 'locate' 'test-localcharset' 'xargs')

# Configs
arm_configs=('arm-clang-12-O0')
x86_configs=('x86-clang-12-O0')

projects=('coreutils_test_list[@]' 'diffutils_test_list[@]' 'findutils_test_list[@]')

function get_unk() {
	base_bins_path=$1
	config=$2
	bin_paths=()
	for proj in ${projects[@]}; do
		base_proj=${proj%%_*}
		for bin in ${!proj}; do
			bin_paths+=("$base_bins_path/$base_proj/$config/stripped/$bin.out")
		done
	done

	# Activate environment and run VexIR2Vec OOV script
	conda activate vexir
	parallel -j $NUM_THREADS -k python ${VEXIR2VEC_PATH}/vexir2vec-collect-oov-count.py -b {} -o $VEXIR_OUTPUT_FILE -v ${VOCAB_PATH} -d ::: ${bin_paths[@]}
	conda deactivate

	rm -f *.data
}
export -f get_unk

if [[ $base_bins_path == *"arm"* ]]; then
	for config in ${arm_configs[@]}; do
		get_unk $base_bins_path $config
	done
elif [[ $base_bins_path == *"x86"* ]]; then
	for config in ${x86_configs[@]}; do
		get_unk $base_bins_path $config
	done
else
	echo "Invalid Architecture"
fi
