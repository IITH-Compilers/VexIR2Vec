# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Implementation of losses"""

import torch
import torch.nn.functional as F


class contrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, temperature=1.0):
        super(contrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, dist, label):

        loss_contrastive = torch.mean(
            1 / 2 * (label) * torch.pow(dist, 2)
            + 1 / 2 * (1 - label) * torch.pow(F.relu(self.temperature - dist), 2)
        )

        return loss_contrastive


class tripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    Label is of the form (label_output1, label_output2) where label_output1 says if anchor and output1 are similar, same for label_output2.
    """

    def __init__(self, temperature=2.0):
        super(tripletLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = (anchor - positive).pow(2).sum(1)

        squarred_distance_2 = (anchor - negative).pow(2).sum(1)

        triplet_loss = F.relu(
            self.temperature + squarred_distance_1 - squarred_distance_2
        ).mean()

        return triplet_loss
