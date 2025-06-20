# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(
        self,
        model=None,
        data_loader=None,
        train_times=1000,
        alpha=0.5,
        use_gpu=True,
        opt_method="sgd",
        save_steps=None,
        checkpoint_dir=None,
    ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

    def trainOneStep(self, data):
        self.optimizer.zero_grad()
        loss = self.model(
            {
                "batch_h": self.toVar(data["batch_h"], self.use_gpu),
                "batch_t": self.toVar(data["batch_t"], self.use_gpu),
                "batch_r": self.toVar(data["batch_r"], self.use_gpu),
                "batch_y": self.toVar(data["batch_y"], self.use_gpu),
                "mode": data["mode"],
            }
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        # training_range = tqdm(range(self.train_times))
        logfile = os.path.join(self.checkpoint_dir, "transe.log")
        logging.basicConfig(filename=logfile, filemode="w", format="%(message)s")
        losses = []

        for epoch in range(self.train_times):
            res = 0.0
            for data in self.data_loader:
                loss = self.trainOneStep(data)
                res += loss
            losses.append(res)
            # training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            print("Epoch %d | loss: %f" % (epoch, res))
            logging.error("Epoch %d | loss: %f" % (epoch, res))

            if (
                self.save_steps
                and self.checkpoint_dir
                and (epoch + 1) % self.save_steps == 0
            ):
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "E" + str(epoch) + ".ckpt")
                )

        plt.plot(losses, label="Loss")
        plt.legend()
        plt.savefig(self.checkpoint_dir + "/loss_plot.png")
        plt.close()

    def setModel(self, model):
        self.model = model

    def toVar(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def setUseGpu(self, use_gpu):
        self.use_gpu = use_gpu

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setLrDecay(self, lr_decay):
        self.lr_decay = lr_decay

    def setWeightDecay(self, weight_decay):
        self.weight_decay = weight_decay

    def setOptMethod(self, opt_method):
        self.opt_method = opt_method

    def setTrainTimes(self, train_times):
        self.train_times = train_times

    def setSaveSteps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.setCheckpointDir(checkpoint_dir)

    def setCheckpointDir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
