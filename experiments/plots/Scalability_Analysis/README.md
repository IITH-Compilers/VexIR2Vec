## Scalability Analysis
<a name="scalability-analysis-plots"></a>

`VexIR2Vec` supports scalable embedding generation through both thread-level (parallel function processing) and task-level (parallel binary processing) parallelism. This design enables efficient handling of large binary datasets.

### Prerequisites
- List of binary paths (filtered set)
- CSV file with: `<binary path>, <binary size in KB>, <function count>`

### Steps
- Run `real-time-extractor.py` to extract `VexIR2Vec` processing times
- Run `norm-time-calc.py` to compute normalization times
- Run `norm_time_vs_total_time.py` to plot normalization vs total time
- Run `time_vs_size.py` to plot embedding time across thread levels (1, 2, 4, 8)
