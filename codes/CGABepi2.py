#coding=utf-8
import warnings
warnings.filterwarnings("ignore")
import random
import torch.nn as nn

from data_encode import *

USE_CUDA = torch.cuda.is_available()

random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)

threshold = 0.5
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EMBED_SIZE = 6
K_Fold = 5

import retnet
layers = 24
hidden_dim = 256
ffn_size = 512
heads = 8

NUM_WORKERS = 4

class CBR(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.CBR(x)

class maxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2)
    def forward(self, x):
        x = self.maxpool(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class CNNEncoder(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()
        self.cbr = CBR(n_channels, out_channels)
        self.maxp = maxpool()

    def forward(self, x):
        x = self.cbr(x)
        x = self.maxp(x)
        return x


class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()
        self.cnn_encoder1 = CNNEncoder(28, 64)
        self.cnn_encoder2 = CNNEncoder(32, 256)
        self.gru1 = nn.GRU(128, 128,
                           batch_first=True,
                           bidirectional=True,
                           num_layers=5
                           )
        self.attn = nn.MultiheadAttention(256, 4)
        self.full_conn = nn.Sequential(
            nn.Linear(256 * 25, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, peps3):
        x = peps3.permute(0, 2, 1)
        x = self.cnn_encoder1(x)
        x = self.cnn_encoder2(x)
        x = x.permute(0, 2, 1)
        x = self.gru1(x)[0]
        x = self.full_conn(torch.flatten(x, start_dim=1))
        return x

