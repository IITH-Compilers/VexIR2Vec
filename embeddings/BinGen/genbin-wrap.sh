# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# Generate binaries for projects

# Usage: bash genbin-wrap.sh <NUM_THREADS>
# ex: bash genbin-wrap.sh 36

NUM_THREADS=$1

bash generateBinaries.sh /path/to/coreutils-9.0 /path/to/binaries/coreutils $NUM_THREADS
bash generateBinaries.sh /path/to/diffutils-3.8 /path/to/binaries/diffutils $NUM_THREADS
bash generateBinaries.sh /path/to/findutils-4.9.0 /path/to/binaries/findutils $NUM_THREADS
bash generateBinaries.sh /path/to/lua-source /path/to/binaries/lua $NUM_THREADS
bash generateBinaries.sh /path/to/gzip-1.12 /path/to/binaries/gzip $NUM_THREADS
bash generateBinaries.sh /path/to/putty-0.76 /path/to/binaries/putty $NUM_THREADS
bash generateBinaries.sh /path/to/curl-7.83.0 /path/to/binaries/curl $NUM_THREADS
