# Generation of Seed Embedding Vocabulary
This directory helps in generating seed embedding vocabulary in 2 steps.
1. [Generating Triplets](#step-1-generating-triplets)
2. [Training TransE to generate seed embedding vocabulary](#step-2-training-transe-to-generate-seed-embedding-vocabulary)

To generate seed embeddings use the conda environment defined in [`openke.yml`](./seed_embeddings/openke.yml)

## Step 1: Generating Triplets

[genAngrTriplets.sh](./seed_embeddings/genAngrTriplets.sh) wrapper script executes the main `driver.py` with the `-t` flag to generate triplets.
```
bash genAngrTriplets.sh <Num-THREADS> <Dir-containing-all-the-binaries> <Dest-dir>
```

## Step 2: Training TransE to generate seed embedding vocabulary
The [`OpenKE`](./seed_embeddings/openke) directory is a modified version of [OpenKE repository](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch) with the necessary changes for training seed embedding vocabulary.

Please see [openke/README.md](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/README.md) for further information on OpenKE.

### Generation of `relation2id.txt`, `entity2id.txt` and `train2id.txt`:

Run [preprocess.py](./seed_embeddings/preprocess.py)
```
python preprocess.py --tdir=/path/to/generated/triplets --odir=/path/to/store/.txt/files
```

### Creation of the seed embeddings

Run [train_transe.py](./seed_embeddings/train_transe.py) to train the OpenKE model with
preprocessed entities obtained from the triplets.
```
python train_transe.py --fpath /path/to/.txt/files --temperature 3 --epochs 600 --lr 0.001 --dim 128 --opt adam --ckpt_dir /path/to/store/embedding
```
