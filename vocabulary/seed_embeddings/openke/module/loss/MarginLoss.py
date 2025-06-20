import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss


class MarginLoss(Loss):
    def __init__(self, adv_temperature=None, temperature=6.0):
        super(MarginLoss, self).__init__()
        self.temperature = nn.Parameter(torch.Tensor([temperature]))
        self.temperature.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def getWeights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (
                self.getWeights(n_score)
                * torch.max(p_score - n_score, -self.temperature)
            ).sum(dim=-1).mean() + self.temperature
        else:
            return (
                torch.max(p_score - n_score, -self.temperature)
            ).mean() + self.temperature

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()
