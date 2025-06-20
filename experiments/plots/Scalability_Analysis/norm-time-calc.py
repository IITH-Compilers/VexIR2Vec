# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Scalability Analysis Script for Normalization Time vs Total Time Plotting"""

# For consolidating norm times of each binary of scalability analysis

import os

base_dir = "/path/to/norm_times"  # it contains norm2 times
outFile = "/path/to/output_file.txt"  # output file to write the total times

ofile = open(outFile, "a")

for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    print(item_path)

    total_time = 0.0

    # Open the file and read line by line
    with open(item_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:
                _, time_str = line.split(
                    "\t"
                )  # Split the line using tab ('\t') as the delimiter
                # print(time_str)
                # exit()
                time = float(time_str)  # Convert the time from string to float
                total_time += time  # Accumulate the time

    # print("Total time:", total_time)
    ofile.write(str(total_time))
    ofile.write("\n")

ofile.close()
