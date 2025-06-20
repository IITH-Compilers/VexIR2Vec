# GroundTruth

The ground truth for the binary diffing experiment consists of function pairs from stripped source and target binaries, compiled from the same source code using different configurations. To identify matching functions, we first align stripped binaries with their unstripped versions using function addresses. Then, we extract symbol and debug info from the unstripped binaries. Function pairs with matching source filenames and function names are considered identical and added to the ground truth.

### Groundtruth Generation

```
bash testset-gen-wrapper.sh <Project-name> <Path to x86 data> <Path to ARM data> <Output path> <NUM_THREADS>
```
**NOTE:**  `test-set-gen-nolib.py` must be present in same working directory.

###  For the generation of ground truth for searching, please refer to [`./BinSearch`](./BinSearch).
