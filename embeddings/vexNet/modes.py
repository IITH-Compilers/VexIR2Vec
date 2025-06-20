# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.


"""Collection of helper functions in the pipeline"""

import os
import sys
import time
import h5py
import torch
import pickle
import random
import tempfile
import statistics
import numpy as np
import pandas as pd
import multiprocessing
from utils import NUM_SB, INP_DIM, OUT_DIM, SEED, savePlot
from utils_inference import kdtreeNeigh
from online_triplet_loss.losses import *
from scipy.spatial import cKDTree

# from models import FloodFCwithAttn, SiameseNetwork
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ray import train
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import BCELoss
from ray.train import Checkpoint
from concurrent.futures import ThreadPoolExecutor, as_completed

from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import distances, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import regularizers

sys.path.append(os.path.abspath("../../experiments/diffing"))
from v2v_diffing import getScores
from utils_inference import kdtreeNeigh

# from ray.air.checkpoint import Checkpoint
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance(p=2)
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
np.set_printoptions(threshold=sys.maxsize)
NUM_NEIGHBORS = 10


DATA_PATH = "/path/to/project/datafiles"

GROUND_TRUTH_CSV = "/path/to/groundtruth.csv"


def contrastiveForward(data, device, model, lossFn, criterion, use_cfg):
    fnEmbed_x, strEmbed_x, libEmbed_x, fnEmbed_y, strEmbed_y, libEmbed_y, label = data
    strEmbed_x, strEmbed_y, label = (
        strEmbed_x.float().to(device),
        strEmbed_y.float().to(device),
        label.float().to(device),
    )
    libEmbed_x, libEmbed_y = libEmbed_x.float().to(device), libEmbed_y.float().to(
        device
    )
    fnEmbed_x = fnEmbed_x.float().view(-1, 1, NUM_SB, INP_DIM).to(device)
    fnEmbed_y = fnEmbed_y.float().view(-1, 1, NUM_SB, INP_DIM).to(device)

    pred1 = model(fnEmbed_x, strEmbed_x, libEmbed_x)
    pred1 = pred1.view(-1, OUT_DIM)
    pred2 = model(fnEmbed_y, strEmbed_y, libEmbed_y)
    pred2 = pred2.view(-1, OUT_DIM)

    if lossFn == "cosine":
        label[label == 0] = -1
        loss = criterion(pred1, pred2, label)
    else:
        loss = criterion(pdist(pred1, pred2), label)
    return loss


def offlineTripletForward(data, device, model, criterion, use_cfg):
    fnEmbed_x, fnEmbed_y, fnEmbed_z, poslabel, negLabel = data

    fnEmbed_x, fnEmbed_y, fnEmbed_z = (
        fnEmbed_x.float().to(device),
        fnEmbed_y.float().to(device),
        fnEmbed_z.float().to(device),
    )
    fnEmbed_x, fnEmbed_y, fnEmbed_z = (
        fnEmbed_x.view(-1, 1, NUM_SB, INP_DIM),
        fnEmbed_y.view(-1, 1, NUM_SB, INP_DIM),
        fnEmbed_z.view(-1, 1, NUM_SB, INP_DIM),
    )

    output1 = model(fnEmbed_x)
    output2 = model(fnEmbed_y)
    output3 = model(fnEmbed_z)
    loss = criterion(output1, output2, output3)
    return loss


def onlineTripletForward(data, device, model, temperature, use_cfg, epoch):
    if use_cfg:
        labels, embeddings, strEmbed, libEmbed, cfg = data
        cfg = cfg.float().to(device)
    else:
        labels, O_emb, T_emb, A_emb, strEmbed, libEmbed = data
        cfg = None
    O_emb = O_emb.view(-1, INP_DIM).float().to(device)
    T_emb = T_emb.view(-1, INP_DIM).float().to(device)
    A_emb = A_emb.view(-1, INP_DIM).float().to(device)
    strEmbed = strEmbed.float().to(device)
    libEmbed = libEmbed.float().to(device)

    outputs, attn_weights = model(O_emb, T_emb, A_emb, strEmbed, libEmbed)

    outputs = outputs.view(-1, OUT_DIM)

    k = 35
    hard_pairs = miners.BatchEasyHardMiner(neg_strategy="hard")(outputs, labels)

    loss = pml_losses.NTXentLoss(temperature=temperature)(outputs, labels, hard_pairs)
    return loss, outputs, attn_weights


def cosFaceForward(data, device, model, temperature, num_classes):
    labels, embeddings, strEmbed, libEmbed = data

    embeddings = embeddings.float().view(-1, 1, NUM_SB, INP_DIM).to(device)
    strEmbed = strEmbed.float().to(device)
    libEmbed = libEmbed.float().to(device)

    outputs = model(embeddings, strEmbed, libEmbed, cfg=None)
    outputs = outputs.view(-1, OUT_DIM)

    loss = pml_losses.CosFaceLoss(
        num_classes=num_classes,
        embedding_size=OUT_DIM,
        temperature=temperature,
        scale=64,
    )(outputs, labels)

    return loss, outputs


def forward(args, data, device, model, criterion, epoch, classes):
    # Contrastive loss
    if args.loss == "cont" or args.loss == "cosine":
        loss = contrastiveForward(
            data, device, model, args.loss, criterion, args.use_cfg
        )

        # Offline Triplet loss
    elif args.loss == "trp":
        loss = offlineTripletForward(data, device, model, criterion, args.use_cfg)

    elif args.loss == "cosface":
        loss, outputs = cosFaceForward(
            data, device, model, args.temperature, num_classes=len(classes)
        )

        # Online Triplet loss
    else:
        loss, outputs, attn_weights = onlineTripletForward(
            data, device, model, args.temperature, args.use_cfg, epoch
        )

    return loss, outputs, attn_weights


def computeNdcg(relevent_indices, len):
    total_dcg = 0
    idcg = 0.0
    for i in range(len):
        if i in relevent_indices:
            total_dcg += 1.0 / np.log2(i + 2)
        idcg += 1.0 / np.log2(i + 2)
    return total_dcg / idcg


def computeAveragePrecision(relevant_indices, len):
    num_relevant = relevant_indices.shape[0]
    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    num_retrieved_relevant = 0

    for i in range(len):
        if i in relevant_indices:
            num_retrieved_relevant += 1
            precision = num_retrieved_relevant / (i + 1)
            precision_sum += precision

    average_precision = precision_sum / num_relevant

    return average_precision


def processQuery(idx, search_embeds, search_keys, kdt):
    query_embedding = search_embeds[idx]
    query_key = search_keys[idx]

    vexir2vec_func_index, vexir2vec_func_dist = kdt.getTopkNeighAndDist(
        query_embedding, len(search_keys), val=True
    )

    # Remove the query key from the list of nearest neighbors
    vexir2vec_func_index = np.delete(
        vexir2vec_func_index, np.where(vexir2vec_func_index == idx)
    )
    vexir2vec_func_dist = np.delete(
        vexir2vec_func_dist, np.where(vexir2vec_func_index == idx)
    )

    relevant_indices = np.where(search_keys[vexir2vec_func_index] == query_key)[0]

    average_precision = computeAveragePrecision(
        relevant_indices, len(vexir2vec_func_index)
    )

    return average_precision


def batchProcessQueries(batch_indices, search_embeds, search_keys, kdt):
    batch_results = []
    for idx in batch_indices:
        query_embedding = search_embeds[idx]
        query_key = search_keys[idx]

        vexir2vec_func_index, vexir2vec_func_dist = kdt.getTopkNeighAndDist(
            query_embedding, len(search_keys)
        )

        vexir2vec_func_index = np.delete(
            vexir2vec_func_index, np.where(vexir2vec_func_index == idx)
        )
        vexir2vec_func_dist = np.delete(
            vexir2vec_func_dist, np.where(vexir2vec_func_index == idx)
        )

        relevant_indices = np.where(search_keys[vexir2vec_func_index] == query_key)[0]
        ap_list = computeAveragePrecision(relevant_indices, len(vexir2vec_func_index))
        ndcg_list = computeNdcg(relevant_indices, len(vexir2vec_func_index))

        batch_results.append((ap_list, ndcg_list))

    return batch_results


def searchOnValDataset(search_key_embed_list, num_threads=24, batch_size=100):
    search_keys, search_embeds = zip(*search_key_embed_list)
    search_embeds = np.array(search_embeds)
    search_keys = np.array(search_keys)

    kdt = kdtreeNeigh(search_embeds)

    num_queries = len(search_keys)
    num_batches = (num_queries + batch_size - 1) // batch_size

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_queries)
            batch_indices = range(start_idx, end_idx)
            future = executor.submit(
                batchProcessQueries, batch_indices, search_embeds, search_keys, kdt
            )
            futures.append(future)

        batch_results = [
            result for future in as_completed(futures) for result in future.result()
        ]

    mean_ap = round(np.mean([result[0] for result in batch_results]), 3)
    mean_ndcg = round(np.mean([result[1] for result in batch_results]), 3)

    return mean_ap, mean_ndcg


def trainSiamese(
    args,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    scheduler,
    use_classifier_head=False,
    classes=None,
):
    # print(args.best_model_path)
    # exit(0)

    device = args.device
    c = 0
    counter = []
    loss_history = []
    avg_losses = []
    avg_val_losses = []
    best_val = 1000000
    best_epoch = 1
    avg_loss_best_val = 0
    if not os.path.exists("./lossplots"):
        os.mkdir("./lossplots")
    filename = "{}_lossplot.png".format(args.config_name)
    print("Model on GPU? ", next(model.parameters()).is_cuda)
    loss_func = pml_losses.TripletMarginLoss()

    input_gt_directory = "/path/to/sample-project-ground-truth-csv"

    projects = []
    csvs_list = []

    if args.findutils:
        projects.append("findutils")
    if args.diffutils:
        projects.append("diffutils")
    if args.coreutils:
        projects.append("coreutils")
    if args.curl:
        projects.append("curl")
    if args.lua:
        projects.append("lua")
    if args.putty:
        projects.append("putty")
    if args.gzip:
        projects.append("gzip")

    print(f"Projects considered for validation set : {projects }")

    subdirectories = [
        "exp1-x86-sb",
        "exp1-arm-sb",
        "exp2-x86-sb",
        "exp2-arm-sb",
        "exp3-x86-arm-sb",
        "exp4-x86-arm-sb",
        "exp4-x86-sb",
    ]

    chosen_csvs = {}

    def chooseCsv(project, subdirectory):
        if project not in chosen_csvs:
            chosen_csvs[project] = {}
        if subdirectory not in chosen_csvs[project]:
            csv_files = os.listdir(
                os.path.join(
                    input_gt_directory,
                    f"{project }-ground-truth-csv",
                    subdirectory,
                )
            )
            chosen_csvs[project][subdirectory] = random.choice(csv_files)
        return chosen_csvs[project][subdirectory]

        # Initialize csvs_list to store selected csv files

    csvs_list = []

    def calculateScores(tp, fp, fn):
        if tp != 0 or fp != 0:
            prec = round(tp / (tp + fp), 3)
        else:
            prec = "NA"

        if tp != 0 or fn != 0:
            rec = round(tp / (tp + fn), 3)
        else:
            rec = "NA"

        if prec != "NA" and rec != "NA" and (prec != 0 or rec != 0):
            f1 = round((2 * prec * rec) / (prec + rec), 3)
        else:
            f1 = "NA"
        return prec, rec, f1

    for project in projects:

        for subdirectory in subdirectories:

            selected_csv = chooseCsv(project, subdirectory)

            csvs_list.append(
                os.path.join(
                    input_gt_directory,
                    f"{project }-ground-truth-inline-sub-fixed-filter",
                    subdirectory,
                    selected_csv,
                )
            )

            # Print the list of selected CSV files
    print(csvs_list)

    for epoch in range(1, args.epochs + 1):
        losses = []
        val_losses = []
        attn_wts_epoch = [0, 0, 0]
        start = time.time()
        model.train()
        for _, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            # print(data)
            loss, outputs, attn_weights = forward(
                args, data, device, model, criterion, epoch, classes
            )

            # change 3xBSx1 shape to BSx3 and obtaining average 1x3
            attn_weights = attn_weights.squeeze().T
            attn_weights = attn_weights.detach().cpu().numpy()
            attn_weights = np.mean(attn_weights, axis=0)

            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            losses.append(loss.item())

            attn_wts_epoch[0] = attn_wts_epoch[0] + attn_weights[0]
            attn_wts_epoch[1] = attn_wts_epoch[1] + attn_weights[1]
            attn_wts_epoch[2] = attn_wts_epoch[2] + attn_weights[2]

            counter.append(c)
            c += 1
        counter.append(epoch - 1)
        avg_loss = round(sum(losses) / len(losses), 5)
        avg_losses.append(avg_loss)

        attn_wts_epoch = [round(i / len(losses), 2) for i in attn_wts_epoch]

        model.eval()
        all_val_outputs = torch.empty((1024, OUT_DIM), device=device)

        total_key_embed_list = []
        total_keys = []

        with torch.no_grad():
            for _, data in enumerate(val_dataloader):
                loss, outputs, attn_weights = forward(
                    args, data, device, model, criterion, epoch, classes
                )
                labels, _, _, _, _, _ = data

                # Convert tensor labels to list of keys
                keys = labels.tolist()
                total_keys.extend(keys)

                # Extend total_key_embed_list with embeddings zipped with respective keys
                total_key_embed_list.extend(zip(keys, outputs.tolist()))

                val_losses.append(loss.item())

                # Concatenate outputs to all_val_outputs
                all_val_outputs = torch.cat((all_val_outputs, outputs), dim=0)

        avg_val_loss = round(sum(val_losses) / len(val_losses), 5)
        map, ndcg = searchOnValDataset(total_key_embed_list)

        precision, recall, F1 = getScores(
            model, DATA_PATH, args.use_cfg, GROUND_TRUTH_CSV, cosine_similarity=False
        )

        avg_val_losses.append(avg_val_loss)
        temp_checkpoint_dir = tempfile.mkdtemp()
        if not args.tune:
            temp_checkpoint_dir = args.best_model_path
        checkpoint = None
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch

            print(
                "Saving model - ",
                os.path.join(temp_checkpoint_dir, args.config_name + ".model"),
            )
            print("temp_checkpoint_dir: ", temp_checkpoint_dir)
            torch.save(
                model,
                os.path.join(temp_checkpoint_dir, args.config_name + ".model"),
            )

            # for saving the details of the model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join(temp_checkpoint_dir, args.config_name + ".all.pt"),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        duration = round(time.time() - start, 2)
        num_op_dups = len(all_val_outputs) - len(torch.unique(all_val_outputs, dim=0))
        map_f1_mean = statistics.mean([map, F1])

        print(
            "Epoch: {}\tLoss: {}\t Val Loss: {}\t Val MAP: {}\t NDCG: {}\t F1: {} \tMAP_F1_Mean: {} \tAttn Weights: {}\t Time taken: {}s".format(
                epoch,
                avg_loss,
                avg_val_loss,
                map,
                ndcg,
                F1,
                map_f1_mean,
                attn_wts_epoch,
                duration,
            )
        )

        if args.tune:
            train.report(
                {
                    "model_path": os.path.join(
                        temp_checkpoint_dir, args.config_name + ".model"
                    ),
                    "pt": os.path.join(
                        temp_checkpoint_dir, args.config_name + ".all.pt"
                    ),
                    "loss": avg_loss,
                    "val_dup": num_op_dups,
                    "NDCG": ndcg,
                    "MAP": map,
                    "F1": F1,
                    "attn_wts": attn_wts_epoch,
                    "total_val": len(all_val_outputs),
                },
                checkpoint=checkpoint,
            )

        scheduler.step()

    return
