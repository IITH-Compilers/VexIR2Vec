import torch
import torch.nn as nn
from .Model import Model
import numpy
from numpy import fft


class HolE(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, temperature=None, epsilon=None):
        super(HolE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.temperature = temperature
        self.epsilon = epsilon
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        if temperature == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.temperature + self.epsilon) / self.dim]),
                requires_grad=False,
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )

    def Conj(self, tensor):
        zero_shape = (list)(tensor.shape)
        one_shape = (list)(tensor.shape)
        zero_shape[-1] = 1
        one_shape[-1] -= 1
        ze = torch.zeros(size=zero_shape, device=tensor.device)
        on = torch.ones(size=one_shape, device=tensor.device)
        matrix = torch.cat([ze, on], -1)
        matrix = 2 * matrix
        return tensor - matrix * tensor

    def Real(self, tensor):
        dimensions = len(tensor.shape)
        return tensor.narrow(dimensions - 1, 0, 1)

    def Imag(self, tensor):
        dimensions = len(tensor.shape)
        return tensor.narrow(dimensions - 1, 1, 1)

    def Mul(self, real_1, imag_1, real_2, imag_2):
        real = real_1 * real_2 - imag_1 * imag_2
        imag = real_1 * imag_2 + imag_1 * real_2
        return torch.cat([real, imag], -1)

    def Ccorr(self, a, b):
        a = self.Conj(torch.rfft(a, signal_ndim=1, onesided=False))
        b = torch.rfft(b, signal_ndim=1, onesided=False)
        res = self.Mul(self.Real(a), self.Imag(a), self.Real(b), self.Imag(b))
        res = torch.ifft(res, signal_ndim=1)
        return self.Real(res).flatten(start_dim=-2)

    def Calc(self, h, t, r, mode):
        if mode != "normal":
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        score = self.Ccorr(h, t) * r
        score = torch.sum(score, -1).flatten()
        return score

    def forward(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self.Calc(h, t, r, mode)
        return score

    def regularization(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3
        return regul

    def l3Regularization(self):
        return (
            self.ent_embeddings.weight.norm(p=3) ** 3
            + self.rel_embeddings.weight.norm(p=3) ** 3
        )

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()
