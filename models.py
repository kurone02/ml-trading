import typing
import pandas as pd
import numpy as np
import numpy.typing as npt
from utils import load_financial_data
from markov import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW
from models import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(42)


class SimpleMLP(nn.Module):

    def __init__(self, 
                 lookback: int=5, 
                 layer_dims: list[int] = []
                ) -> None:
        super().__init__()

        self.linear_layers = []
        prev_features = lookback * len(STATE_TO_NUM)
        for dim in layer_dims:
            self.linear_layers.append(nn.Linear(
                in_features=prev_features,
                out_features=dim,
            ))
            prev_features = dim
        self.linear_layers.append(nn.Linear(
            in_features=prev_features,
            out_features=3,
        ))

        self.linear_layers = nn.Sequential(*self.linear_layers)
        
    def forward(self, x):
        x = self.linear_layers(x)
        x = F.softmax(x, dim=1)
        return x


class SimpleLSTM(nn.Module):

    def __init__(self, 
                 lookback: int=5, 
                 hidden_size: int=16,
                ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.lookback = lookback

        self.lstm = nn.LSTM(
            input_size=len(STATE_TO_NUM),
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.lookback * self.hidden_size,
            out_features=3,
        )
        
    def forward(self, x):
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x.reshape((-1, self.lookback * self.hidden_size))
        # print(x.shape)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        # print(x.shape)
        return x


