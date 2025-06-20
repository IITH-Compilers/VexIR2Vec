# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Real Time Extractor Script for Scalability Analysis"""

# For extracting real times from output script for Scalability Analysis
# Eg usage
# python real-time-extractor.py -ip vexir/x86-stripped-101-plot-filtered-real-times-fchnk1.txt-temp.txt -op vexir/only-times-x86-stripped-101-plot-filtered-real-times-fchnk1.txt


import re
import argparse


def extractRealTimes(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    real_times = []
    for line in lines:
        if line.startswith("real"):
            match = re.search(r"\d+m\d+\.\d+s", line)
            if match:
                real_times.append(match.group())

    with open(output_file, "w") as f:
        for real_time in real_times:
            f.write(real_time + "\n")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-ip",
    "--input file",
    dest="ip_file",
    help="input file name with path",
    default=None,
)
parser.add_argument(
    "-op", "--output file", dest="op_file", help="output file with path", default=None
)
args = parser.parse_args()


input_file = args.ip_file
output_file = args.op_file
extractRealTimes(input_file, output_file)
