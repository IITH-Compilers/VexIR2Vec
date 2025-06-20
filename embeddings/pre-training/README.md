# Data Generation

1. [Without Database](#without-database): This mode skips the database and generates embeddings
   on the fly every time, without saving or reusing any intermediate results.
2. [With Database](#with-database): This mode stores static data that is only needed to be
   computed once—in a database. It also speeds up angr processing using its built-in serialization.

[genData.sh](./embedding-gen/genData.sh) is used for generating embedding files from the processed binaries in both with and without Database flows
using the VexIR2Vec framework.

```
bash genData.sh <path-to-x86-data-all | path-to-arm-data-all> <num-threads> <path-to-seed-embedding> <normalization-level> <num-func-chunks> <mode>
```
Eg. `bash genData.sh ~/x86-data-all/ 5 ~/seed_embedding 3 2 db`

- `num-threads` and `num-func-chunks` determine the degree of parallelism. While `num-threads` parallelizes at a coarser level (number of
binaries processed in parallel), `num-func-chunks` determine parallelism at a finer level (number of functions processed in paralle within a binary).
- `normalization-level` indicates the intensity of normalization that we apply on peepholes. Unless required, choose `3`.
- `mode` can be one of `db` or `non-db`.


## Without Database
Run [genData.sh](./genData.sh) by setting `non-db` as the `mode`

```
bash genData.sh <path-to-binaries> <num-threads> <path-to-seed-embedding> <normalisation> <num-func-chunks> non-db
```
Eg. `bash genData.sh ~/x86-data-all/ 5 ~/seed_embedding 2 2 non-db`

## With Database
This section explains how to generate the necessary database files needed to create the data for the Embedding generation pipeline.

> [!IMPORTANT]
> Binary Path Structure
> Our scripts assume that the binaries are stored in the following directory structure
```
x86-data-all
└── diffutils # Your project name
    └── x86-clang-12-O0 # arch-comp-version-optlevel
        └── unstripped
		└── binary.out
        └── stripped
		└── binary.out
    └──
    └── x86-gcc-6-O0
	└── unstripped
		└── binary.out
	└── stripped
		└── binary.out

arm-data-all
└──
```

> If you had used the provided [binary generation scripts](../BinGen), it would generate the binaries in the above structure.

### 1. MetaData Generation
The [genMetaDataDB.sh](./db-gen/genMetaDataDB.sh) generates necessary metadata from the binaries.

```bash
genMetaDataDB.sh <path-to-x86-data-all | path-to-arm-data-all> <num-threads> <output-db-path>

```

### 2. `Embedding` File Generation
The [genData.sh](./embedding-gen/genData.sh) is used for the **final step** in the pipeline .

```
bash genData.sh <path-to-x86-data-all | path-to-arm-data-all> <num-threads> <path-to-seed-embedding> <embedding-db-path> <normalization-level> <num-func-chunks> db
```
Eg. `bash genData.sh ~/x86-data-all/ 5 ~/seed_embedding ~/embedding.db 3 2 db`

> [!Caution]
> **Repairing Corrupted angrDB**
> If you encounter an error related to a corrupted `angrDB` during data generation:
>   - Delete the corrupted database files.
>   - Regenerate the database from scratch by rerunning the data generation scripts.
>   - Check that the path to the database is correct and points to the intended location.
