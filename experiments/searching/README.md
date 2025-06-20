# Searching

**Searching** is a binary similarity task aimed at retrieving the function from a set that most closely matches a given query function.

#### Key Points:
- Let **A** = {a₁, a₂, ..., aₘ} be a collection of VexIR2Vec vectors from various binaries.
- Let **b** be a VexIR2Vec vector representing the query function.
- The search module `Msearch(A, b)` returns a single vector **aᵢ** from **A**.
- The goal is to find the function **aᵢ** in **A** that performs the **same task** as the function represented by **b**.
- This task is analogous to nearest-neighbor retrieval in an embedding space.

- Example Usage:

  ```
  python3 v2v_search_wrapper.py -bmp /path/to/.model -dp /path/to/x86-data-all -search_gt_dir /path/to/GroundTruth -res_dir /path/to/store_results -out_dir /path/to/store_roc -n <threads> -chunks <num_chunks> -filter <projects-needed> -config <config-needed>

  ```

- Key Parameters for `v2v_search_wrapper.py`

| Parameter        | Description                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-bmp`           | Path to the trained model (Base Model Path)                                                                                                       |
| `-dp`            | Directory containing data files (embeddings to be searched)                                                                                       |
| `-search_gt_dir` | Directory containing ground truth for evaluation                                                                                                  |
| `-out_dir`       | Output directory where search results will be saved                                                                                               |
| `-res_dir`       | Directory to store evaluation/comparison results (e.g., for diffing)                                                                              |
| `-n`             | Number of threads                                                                                     |
| `-chunks`        | Number of chunks to split the search into                                                                                                         |
| `-filter`        | Filter to restrict evaluation to a specific project; choose from:<br> `["findutils", "diffutils", "coreutils", "gzip", "lua", "curl", "putty"]`   |
| `-config`        | Filter to restrict evaluation to a specific compiler configuration; choose from:<br> `["x86-clang-8", "x86-clang-12", "x86-gcc-8", "x86-gcc-10"]` |
