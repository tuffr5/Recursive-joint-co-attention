import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    """
    normalize features
    """
    def __init__(self):
        super(Normalization, self).__init__()

    def forward(self, audio, video):

        return F.normalize(audio, dim=-1), F.normalize(video, dim=-1)