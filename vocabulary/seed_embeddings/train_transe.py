# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""This script is used to train and obtain the seed embedding"""

# Usage: python train_transe.py --fpath /path/to/.txt/files --temperature 3 --epochs 600 --lr 0.001 --dim 128 --opt adam --ckpt_dir /path/to/store/embedding

import openke
import os
import argparse
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss, SigmoidLoss, SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
parser.add_argument("--temperature", type=int, required=True, default=3)
parser.add_argument("--epochs", type=int, required=True, default=600)
parser.add_argument("--lr", type=float, required=True, default=0.001)
parser.add_argument("--dim", type=int, required=True, default=128)
parser.add_argument("--opt", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=True)
config = parser.parse_args()


config.fpath = os.path.abspath(config.fpath)
print(config.fpath)
config.ckpt_dir = os.path.abspath(config.ckpt_dir) + "/ckpt_{}M_{}E_{}D_{}LR_{}".format(
    config.temperature, config.epochs, config.dim, config.lr, config.opt
)
try:
    os.makedirs(config.ckpt_dir)
except Exception as err:
    print(err)

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=config.fpath + "/",
    batch_size=512,
    nbatches=256,
    threads=8,
    sampling_mode="normal",
    bern_flag=0,
    filter_flag=1,
    neg_ent=1,
    neg_rel=0,
)


# define the model
transe = TransE(
    ent_tot=train_dataloader.getEntTot(),
    rel_tot=train_dataloader.getRelTot(),
    dim=config.dim,
    p_norm=1,
    norm_flag=True,
)


# define the loss function
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(temperature=config.temperature),
    batch_size=train_dataloader.getBatchSize(),
)

# train the model
print("ckpt_dir: ", config.ckpt_dir)
trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=config.epochs,
    alpha=config.lr,
    use_gpu=True,
    checkpoint_dir=config.ckpt_dir,
    save_steps=500,
    opt_method=config.opt,
)
trainer.run()
transe.saveCheckpoint(os.path.join(config.ckpt_dir, "transe.ckpt"))


rep = transe.ent_embeddings.weight.data
f = open(os.path.join(config.fpath, "entity2id.txt"))
content = f.read()
f.close()

print(config)
seedFile = os.path.join(
    config.ckpt_dir,
    "seedEmbedding_{}M_{}E_{}D_{}LR_{}".format(
        config.temperature, config.epochs, config.dim, config.lr, config.opt
    ),
)

f = open(seedFile, "w")
entities = content.split("\n")
toTxt = ""

for i in range(1, int(entities[0])):
    toTxt += entities[i].split("\t")[0] + ":" + str(rep[i - 1].tolist()) + ",\n"
toTxt += (
    entities[int(entities[0])].split("\t")[0]
    + ":"
    + str(rep[int(entities[0]) - 1].tolist())
)
f.write(toTxt)
