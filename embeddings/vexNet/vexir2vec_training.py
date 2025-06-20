# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

""""Training script to train the model"""

# Usage:
# python vexir2vec_training.py \
#     --dataset_type <str: 'h5'> \
#     --loss <str: loss function> \
#     --optimizer <str: optimizer name> \
#     --beta <float: optimizer beta parameter> \
#     --lr <float: learning rate> \
#     --sched <str: scheduler type> \
#     --gamma <float: scheduler decay rate> \
#     --batch_size <int: batch size> \
#     --epochs <int: number of epochs> \
#     --temperature <float: temperature for loss> \
#     --inp_dim <int: input dimension> \
#     --out_dim <int: output dimension> \
#     --best_model_path <str: path to save model> \
#     --data_path <str: path to data directory> \
#     --test_path <str: path to test directory> \
#     --pretrained_model <str: path to pretrained model> \
#     --use_cfg <bool: use CFG if True> \
#     --tune <bool: use ray tuner if True> \
#     --mperclass <int: m per class sampler value>

import os
import h5py
import time
import random
import math
import numpy as np
from losses import *
from torch import optim
import shutil
from torch.nn import BCELoss
from modes import trainSiamese
from model_OTA import FCNNWithAttention

from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import (
    printVex2vec,
    preprocessH5Dataset,
    genTrainDataPairsNew,
    preprocessH5Cfg,
    modifyKey,
)
from utils import (
    KEYS_FILE,
    SEED,
    INP_DIM,
    NUM_SB,
    TRAIN_DATA_FILE,
    CNN_INP_DIM,
    NUM_WALKS,
)

TRAIN_DATA_FILE_CFG_dir = ""
TRAIN_DATA_H5FILE_DIR = ""
from torch.utils.data import Dataset

from ray.tune.search.optuna import OptunaSearch
import re
import json

import torch.nn as nn

from pytorch_metric_learning import (
    distances,
    miners,
    reducers,
    testers,
    samplers,
    trainers,
)
import ray
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
le = LabelEncoder()
hf = h5py.File(KEYS_FILE, "r")
all_keys = hf.get("keys")[()]
all_keys = modifyKey(all_keys)

le.fit(all_keys)
classes = le.classes_


ray.tune.logger.DEFAULT_LOGGERS = ("ray", "tune", "tensorboard")


class cfgDataset(Dataset):
    def __init__(self, data):
        self.key = data[0]
        self.embed = data[1]
        self.strEmbed = data[2]
        self.libEmbed = data[3]
        self.cfg = data[4]

    def __len__(self):
        return len(self.key)

    def padAndSlice(self, x, CNN_INP_DIM):
        global CFG_str
        global CFG_non_str
        # Pad the input matrix with 0s if its size is less than the threshold size
        try:
            if isinstance(x, str):
                x = np.array(json.loads(x), dtype=float)

            if x.shape[0] < CNN_INP_DIM:
                pad_size = CNN_INP_DIM - x.shape[0]
                x = np.pad(
                    x, ((0, pad_size), (0, pad_size)), mode="    model.eval()constant"
                )

                # Reshape the input matrix if its size is greater than the threshold size
            if x.shape[0] > CNN_INP_DIM:
                new_shape = (CNN_INP_DIM, CNN_INP_DIM)
                x = x[: new_shape[0], : new_shape[1]]
        except Exception as e:

            x = np.zeros((CNN_INP_DIM, CNN_INP_DIM))

        return x

    def __getitem__(self, idx):
        key = torch.tensor(self.key[idx])
        embed = torch.tensor(self.embed[idx])
        strEmbed = torch.tensor(self.strEmbed[idx])
        libEmbed = torch.tensor(self.libEmbed[idx])
        cfg = torch.tensor(
            self.padAndSlice(self.cfg[idx], CNN_INP_DIM), dtype=torch.float
        )
        return key, embed, strEmbed, libEmbed, cfg


def weightsInit(m):
    nn.init.xavier_normal_(m.layer1[0].weight)
    nn.init.xavier_normal_(m.layer2[0].weight)

    print("Initialised linear layer following xavier normal")

    # *********************** VEXIR2Vec Code To Support Ray Tuner ********************************************


def getOptimizer(args, model, config):

    if not config:
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                betas=(args.beta, 0.999),
                lr=args.lr,
                weight_decay=0.01,
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=0.01)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        choice = config["opt"]
        if choice == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                betas=(config["beta"], 0.999),
                lr=config["lr"],
                weight_decay=0.01,
            )
        elif choice == "sgd":
            optimizer = optim.SGD(model.parameters(), config["lr"], weight_decay=0.01)
        else:
            optimizer = optim.RMSprop(
                model.parameters(), lr=config["lr"], weight_decay=0.01
            )
    return optimizer


def getScheduler(args, config, optimizer):
    if not config:
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        scheduler = LinearLR(optimizer)
    else:
        if config["sched"] == "Exp_lr":
            scheduler = ExponentialLR(optimizer, gamma=config["gamma"])
        else:
            scheduler = LinearLR(optimizer)
    return scheduler


def trainVexir2vec(config, args=None):

    device = args.device

    if args.dataset_type == "H5":
        if args.loss == "cont" or args.loss == "cosine":
            train_data, pos_pairs, neg_pairs = genTrainDataPairsNew(TRAIN_DATA_FILE)
            print("Size of Training Data: ", len(train_data))
            print("#Pos pairs: ", len(pos_pairs))
            print("#Neg pairs: ", len(neg_pairs))

            train_size = int(0.8 * len(train_data))
            test_size = len(train_data) - train_size
            print(train_size, test_size)
            train_data, val_data = random_split(train_data, [train_size, test_size])

            train_dataloader = DataLoader(
                train_data,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=10,
            )
            val_dataloader = DataLoader(
                val_data, batch_size=config["batch_size"], shuffle=True, num_workers=10
            )

        else:
            s = time.time()
            print("Loading Training Data ...")
            if args.use_cfg:

                (
                    keys,
                    opcEmbeds,
                    typeEmbeds,
                    argEmbeds,
                    strEmbeds,
                    libEmbeds,
                ) = preprocessH5Dataset(TRAIN_DATA_FILE)
                keys = le.transform(keys)

                cfgs = preprocessH5Cfg(TRAIN_DATA_FILE_CFG_dir)

                dataset = cfgDataset(
                    [keys, opcEmbeds, typeEmbeds, argEmbeds, strEmbeds, libEmbeds, cfgs]
                )
            else:
                (
                    keys,
                    opcEmbeds,
                    typeEmbeds,
                    argEmbeds,
                    strEmbeds,
                    libEmbeds,
                ) = preprocessH5Dataset(TRAIN_DATA_FILE)
                all_valid_keys = keys.copy()
                # Choosing only the unique embeddings
                opcEmbeds, ind = np.unique(opcEmbeds, axis=0, return_index=True)
                keys = keys[ind]
                typeEmbeds = typeEmbeds[ind]
                argEmbeds = argEmbeds[ind]
                strEmbeds = strEmbeds[ind]
                libEmbeds = libEmbeds[ind]

                # Shuffle the data
                length = len(keys)
                shuffled_indices = np.random.permutation(length)

                keys = keys[shuffled_indices]
                typeEmbeds = typeEmbeds[shuffled_indices]
                argEmbeds = argEmbeds[shuffled_indices]
                strEmbeds = strEmbeds[shuffled_indices]
                libEmbeds = libEmbeds[shuffled_indices]

                print("After removing the duplicate embeddings")
                print(f"keys shape: {keys .shape }")
                print(
                    f"Embeds O, T, A shapes : {opcEmbeds .shape }, {typeEmbeds .shape }, {argEmbeds .shape }"
                )
                print(f"strEmbeds shape : {strEmbeds .shape }")
                print(f"libEmbeds shape : {libEmbeds .shape }")

                # choosing only labels which have more than thresh_min datapoints and at most thresh_max samples from each class
                thresh_min = config["thresh_min"]
                thresh_max = config["thresh_max"]
                print(thresh_min, thresh_max)
                uniq_keys, cnts = np.unique(keys, return_counts=True)
                uniq_keys = uniq_keys[cnts >= thresh_min]
                cnt = {}
                ind = []
                for i in range(len(keys)):
                    if keys[i] in uniq_keys:
                        if keys[i] not in cnt:
                            cnt[keys[i]] = 0
                            ind.append(i)
                            cnt[keys[i]] += 1
                        elif cnt[keys[i]] < thresh_max:
                            ind.append(i)
                            cnt[keys[i]] += 1

                keys = keys[ind]

                uniq_keys, cnts = np.unique(keys, return_counts=True)
                opcEmbeds = opcEmbeds[ind]
                strEmbeds = strEmbeds[ind]
                libEmbeds = libEmbeds[ind]
                keys = [k.decode() if isinstance(k, bytes) else k for k in keys]
                all_valid_keys = [
                    k.decode() if isinstance(k, bytes) else k for k in all_valid_keys
                ]
                le = LabelEncoder()
                le.fit(all_valid_keys)
                valid_set = set(le.classes_)
                keys = [k for k in keys if k in valid_set]
                keys = le.transform(keys)

                # Convert keys to a PyTorch tensor
            keys_tensor = torch.from_numpy(keys)

            # Identify unique classes
            classes = torch.unique(keys_tensor)

            # Calculate lengths for train and validation sets
            total_length = len(classes)
            train_length = int(0.8 * total_length)
            val_length = total_length - train_length

            print("Total len: ", total_length)
            print("train_length: ", train_length)
            print("val_length: ", val_length)

            # Split classes into train and validation sets
            train_classes, val_classes = random_split(
                classes, [train_length, val_length]
            )

            # Filter indices corresponding to the train and validation sets
            train_indices = torch.cat(
                [
                    torch.nonzero(keys_tensor == class_label).squeeze()
                    for class_label in train_classes
                ]
            )
            val_indices = torch.cat(
                [
                    torch.nonzero(keys_tensor == class_label).squeeze()
                    for class_label in val_classes
                ]
            )

            # Create train and validation datasets
            train_dataset = TensorDataset(
                keys_tensor[train_indices],
                torch.from_numpy(opcEmbeds)[train_indices],
                torch.from_numpy(typeEmbeds)[train_indices],
                torch.from_numpy(argEmbeds)[train_indices],
                torch.from_numpy(strEmbeds)[train_indices],
                torch.from_numpy(libEmbeds)[train_indices],
            )

            val_dataset = TensorDataset(
                keys_tensor[val_indices],
                torch.from_numpy(opcEmbeds)[val_indices],
                torch.from_numpy(typeEmbeds)[val_indices],
                torch.from_numpy(argEmbeds)[val_indices],
                torch.from_numpy(strEmbeds)[val_indices],
                torch.from_numpy(libEmbeds)[val_indices],
            )

            print(
                "Time taken in data loading and preprocessing: {:.2f}s".format(
                    time.time() - s
                )
            )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=10,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=10,
            )

            args.temperature = config["temperature"]
    else:
        print("Unknown Format. Exiting!")
        exit()

    if args.pretrained_model == "":
        model = FCNNWithAttention(INP_DIM, config=config).to(device)
    else:
        model = torch.load(args.pretrained_model)
        model.to(device)

    if args.loss == "trp":
        criterion = TripletLoss()
    elif args.loss == "cont":
        criterion = ContrastiveLoss(temperature=args.temperature)
    elif args.loss == "cosine":
        criterion = torch.nn.CosineEmbeddingLoss(temperature=args.temperature)
    else:
        criterion = BCELoss()

    optimizer = getOptimizer(args, model, config)

    scheduler = getScheduler(args, config, optimizer)

    print("\nTraining the Siamese Net...")
    trainSiamese(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        scheduler,
        classes=classes,
    )

    print("Training finished.")


def stopFn(trial_id: str, result: dict) -> bool:

    # Stop if 10% of model outputs are duplicates or if training time exceeds 15 seconds per epoch.
    return (
        result["val_dup"] > 0.2 * result["total_val"]
        or result["time_total_s"] > 15 * 100
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    if args.tune:
        ray.init(dashboard_host="0.0.0.0", dashboard_port=8003)

        config = {
            "thresh_min": "choice([int: lower threshold options])",  # e.g., [30, 50, 75, 100, 150, 200]
            "thresh_max": "sample_from(lambda spec: choice([val >= spec.config.thresh_min]))",
            # selects a value >= thresh_min from the same set
            "batch_size": "choice([int: typical batch sizes])",  # e.g., [1024, 2048, 4096]
            "activation": "<str: activation function>",  # e.g., 'relu', 'silu', 'leaky_relu', 'tanh'
            "num_O_layers": "randint(min=1, max=3)",  # number of output layers (inclusive lower, exclusive upper)
            "num_layers": "randint(min=1, max=4)",  # total number of hidden layers
            "hidden": "sample_from(lambda spec: randint(low=100, high=400, size=spec.config.num_layers))",
            # hidden size per layer (int array)
            "concat_layer": "sample_from(lambda spec: randint(low=0, high=spec.config.num_layers))",
            # index for concatenation
            "drop_units": "sample_from(lambda spec: uniform(0.0, 0.3, size=spec.config.num_layers))",
            # dropout rates per layer
            "lr": "loguniform(low=1e-4, high=1e-1)",  # learning rate (log scale)
            "opt": "choice([<str: optimizer names>])",  # e.g., ['adam', 'sgd']
            "beta": "uniform(0.7, 0.99)",  # optimizer-specific beta parameter
            "sched": "choice([<str: lr scheduler types>])",  # e.g., ['Linear_lr', 'StepLR']
            "gamma": "uniform(0.8, 0.99)",  # learning rate decay
            "temperature": "choice([float: temperature values])",
        }

        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=args.epochs,
            grace_period=8,
            reduction_factor=3,
            brackets=1,
            stop_last_trials=False,
        )

        reporter = CLIReporter(
            metric_columns=[
                "loss",
                "training_iteration",
                "val_dup",
                "NDCG",
                "F1",
                "MAP",
            ]
        )

        path = os.path.expanduser("/path/to/checkpoint/")

        if tune.Tuner.can_restore(path):
            tuner = tune.Tuner.restore(
                path,
                trainable=tune.with_resources(
                    tune.with_parameters(trainVexir2vec, args=args),
                    resources={"cpu": 5, "gpu": 0.3},
                ),
                resume_errored=True,
                param_space=config,
            )
            print("Restored from checkpoint")
        else:
            print("No checkpoint to restore from. Starting from scratch.")
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(trainVexir2vec, args=args),
                    resources={"cpu": 5, "gpu": 0.2},
                ),
                tune_config=tune.TuneConfig(
                    search_alg=OptunaSearch(),
                    metric="F1",
                    mode="max",
                    scheduler=scheduler,
                    num_samples=2,
                    reuse_actors=True,
                    max_concurrent_trials=4,
                ),
                run_config=ray.train.RunConfig(
                    progress_reporter=reporter,
                    stop=stopFn,
                    checkpoint_config=ray.train.CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_score_attribute="F1",
                        checkpoint_score_order="max",
                    ),
                ),
                param_space=config,
            )

        results = tuner.fit()

        best_result = results.get_best_result("F1", "max")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final metrics: {}".format(best_result.metrics))
        model_path = best_result.metrics["model_path"]
        pt_path = best_result.metrics["pt"]
        dst_dir = args.best_model_path
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(model_path, os.path.join(dst_dir, args.config_name + ".model"))
        shutil.move(pt_path, os.path.join(dst_dir, args.config_name + ".all.pt"))
        results_df = results.get_dataframe()
        print(results_df.columns)
        filtered_df = results_df[results_df["val_dup"] / results_df["total_val"] <= 0.2]

        # Sort the DataFrame in descending order based on the 'MAP' column
        df_sorted = filtered_df.sort_values(by="F1", ascending=False)

        print("Best trial after filtering the duplicates - ", df_sorted.iloc[0])

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = {
            "activation": "<str: activation function>",  # e.g., 'relu', 'leaky_relu', 'tanh', 'gelu'
            "batch_size": args.batch_size,  # batch size, typically power of 2
            "beta": args.beta,  # e.g., beta1/beta2 for Adam optimizer or loss scaling
            "concat_layer": "<int: 0-num_layers-1>",  # index at which to apply layer concatenation
            "drop_units": "<list[float]: each in 0.0-0.5>",  # dropout rate per layer
            "gamma": "<float: 0.0-1.0>",  # scheduler decay factor or discount rate
            "hidden": "<list[int]: e.g., [64-1024, 64-1024]>",  # hidden units per layer
            "lr": args.lr,  # learning rate
            "temperature": args.temperature,
            "num_O_layers": "<int: 1-3>",  # number of output layers
            "num_layers": "<int: 1-5>",  # number of hidden layers
            "opt": args.optimizer,  # e.g., 'adam', 'sgd', 'adamw'
            "sched": "Linear_lr",  # e.g., 'Linear_lr', 'StepLR', 'CosineAnnealing'
            "thresh_max": "<int: > thresh_min>",  # upper threshold value
            "thresh_min": "<int: < thresh_max>",  # lower threshold value
        }

        trainVexir2vec(config, args=args)


if __name__ == "__main__":

    printVex2vec()
    parser = ArgumentParser(description="VexIR2Vec framework for binary similarity.")
    parser.add_argument("-l", "--loss", required=True, help="Loss to be used.")
    parser.add_argument(
        "-lr", "--lr", required=True, type=float, help="Learning rate to be used."
    )
    parser.add_argument(
        "-b", "--beta", type=float, default=0.9, help="beta1 to be used in Adam."
    )
    parser.add_argument("-bs", "--batch_size", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-temp", "--temperature", type=float, required=True)
    parser.add_argument("-inpd", "--inp_dim", type=int, required=True)
    parser.add_argument("-outd", "--out_dim", type=int, required=True)
    parser.add_argument(
        "-opt", "--optimizer", required=True, help="Optimizer to be used."
    )
    parser.add_argument(
        "-bmp", "--best_model_path", required=True, help="Path to the best model"
    )
    parser.add_argument(
        "-dp", "--data_path", help="Directory containing all the projects."
    )
    parser.add_argument(
        "-testd", "--test_path", help="Directory containing all the projects."
    )
    parser.add_argument("-more", "--more", default="")  # additional remarks
    parser.add_argument("-dt", "--dataset_type", type=str, required=True)
    parser.add_argument(
        "-ptm",
        "--pretrained_model",
        help="Load a pretrained model and train further",
        default="",
    )
    parser.add_argument(
        "-cfg", "--use_cfg", type=bool, default=False, help="Use CFG if set"
    )
    parser.add_argument(
        "-t", "--tune", type=bool, default=False, help="Use to run ray tuner"
    )
    parser.add_argument(
        "-mperc",
        "--mperclass",
        type=int,
        default=100,
        help="value of m in m per class sampler",
    )
    parser.add_argument("-config", "--config", type=str, help="Path to the config")

    args = parser.parse_args()
    args.model = args.loss.lower()
    args.optimizer = args.optimizer.lower()
    args.dataset_type = args.dataset_type.upper()

    args.config_name = "_".join(
        [
            args.more,
            args.loss,
            args.optimizer,
            "lr" + str(args.lr),
            "b" + str(args.batch_size),
            "e" + str(args.epochs),
            "m" + str(args.temperature),
            str(args.inp_dim) + "D",
            str(args.mperclass) + "MperC",
        ]
    )
    if args.beta is not None:
        args.config_name += "_b" + str(args.beta)

    args.best_model_path = args.best_model_path + "_" + args.config_name
    if not os.path.exists(args.best_model_path):
        os.mkdir(args.best_model_path)
    args.best_model_path = os.path.abspath(args.best_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    print(args)
    # exit(0)
    print("Available Device: ", device)
    main(args)
