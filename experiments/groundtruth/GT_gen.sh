# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# VexIR2Vec Groundtruth Generation Script for Diffing Experiments

# Generates the diffing groundtruth csvs
# Usage:
# bash GT_gen.sh

bash testset-gen-wrapper.sh findutils /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh diffutils /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh coreutils /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh lua /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh gzip /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh putty /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
bash testset-gen-wrapper.sh curl /path/to/x86-data-files/ /path/to/arm-data-files/ /path/to/output-groundtruth-diffing/ 24
