# OOV plot
**Dependencies:**
* **VexIR:** vexir2vec-collect-oov-count.py, embeddings_collect_oov_count.py

1. Collecting stats for OOV plot
```
bash collect-oov-wrapper.sh <Path to binaries directory> <Path to VexIR directory> <VexIR OOV output file> <NUM_THREADS>

```
2. Plotting OOV

```
python3 oov-plot.py -vp <Path to vexir2vec oov txt file>

```
