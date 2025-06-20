#!/bin/bash
# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# Generates binaries for provided project and moves to provided output directory

# Usage: bash generateBinaries.sh $PROJECT_PATH $OUTPUT_DIR $NUM_THREADS
# Example: bash generateBinaries.sh /path/to/coreutils-9.0 /path/to/binaries/coreutils $NUM_THREADS

SOURCE_PATH=$1
DEST_PATH=$2
NUM_THREADS=$3
FLAG=$4

if [ -z "$SOURCE_PATH" ]; then
	echo "set SOURCE_PATH as 1st argument"
	exit
fi

if [ -z "$DEST_PATH" ]; then
	echo "set DEST_PATH as 2nd argument"
	exit
fi

archs=('x86')
opts=('O0' 'O1' 'O2' 'O3' 'Os')
compilers=('gcc-6' 'gcc-8' 'gcc-10' 'clang-6' 'clang-8' 'clang-12')

mkdir -p $DEST_PATH
for arch in ${archs[@]}; do
	echo $arch
	for compiler in ${compilers[@]}; do
		echo $compiler
		if [[ "$compiler" == *"clang"* ]]; then
			compiler_cxx="${compiler/clang/clang++}"
		else
			compiler_cxx="${compiler/gcc/g++}"
		fi
		for opt in ${opts[@]}; do
			cd $SOURCE_PATH
			export CC=$compiler
			export CXX=$compiler_cxx
			short_CC=${compiler##*/}
			short_CC=${short_CC%%.*}
			if [ "$FLAG" == "lua" ]; then
				tar -xf lua-5.4.4.tar.gz
				cd lua-5.4.4
				make CC=$compiler CXX=$compiler_cxx CFLAGS="-g -$opt" CXXFLAGS="-g -$opt"
				build_path=$(realpath .)
				conf=$arch-${short_CC}-$opt
			else
				cd $SOURCE_PATH
				export CC=$compiler CFLAGS="-g -$opt"
				export CXX=$compiler_cxx CXXFLAGS="-g -$opt"
				echo $CC
				echo $CXX
				echo $CFLAGS
				echo $CXXFLAGS
				conf=$arch-${short_CC}-$opt
				if [ ! -f "$conf"/Makefile ]; then
					rm -rf $conf
					mkdir $conf && cd $conf
					build_path=$(realpath .)
					if [ "$FLAG" == "curl" ]; then
						$SOURCE_PATH/configure --without-ssl --without-zlib
					else
						$SOURCE_PATH/configure
					fi

					make -j$NUM_THREADS
					cd -
				fi
			fi
			dest="$DEST_PATH"/"$conf"
			mkdir -p $dest/unstripped
			bin_list=$(find $build_path -type f -executable ! -name "*.*" -exec grep -IL . "{}" \;)
			cp $bin_list $dest/unstripped
			find $dest/unstripped -type f -executable -exec mv {} {}.out \;
			if [ "$build_path" != "" ]; then
				rm -rf $build_path
			fi
			mkdir -p $dest/stripped

			for file in $dest/unstripped/*; do
				base=$(basename $file)
				strip -s $file -o $dest/stripped/$base
			done
		done
	done
done
