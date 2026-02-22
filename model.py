import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class FinSeriesDataset(Dataset):
    def __init__(self, parquet_path):
        super().__init__()
        df = pd.read_parquet(parquet_path)
        df["step_in_seq"] = df["step_in_seq"].astype(int)
        df = df.sort_values(["seq_ix", "step_in_seq"])

        feature_cols = (
            [f"p{i}" for i in range(12)]
            + [f"v{i}" for i in range(12)]
            + [f"dp{i}" for i in range(4)]
            + [f"dv{i}" for i in range(4)]
        )
        target_cols = ["t0", "t1"]

        num_seqs = len(df) // 1000

        x_data = df[feature_cols].values.astype(np.float32)
        y_data = df[target_cols].values.astype(np.float32)

        self.x = torch.tensor(x_data).view(num_seqs, 1000, 32)
        self.y = torch.tensor(y_data).view(num_seqs, 1000, 2)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.01):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.proj(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x + residual


class TradingModel(nn.Module):
    def __init__(
        self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=2, noise_std=0.02
    ):
        super().__init__()
        self.noise_std = noise_std
        self.deterministic_mode = False

        self.price_indices = list(range(0, 12)) + list(range(24, 28))
        self.vol_indices = list(range(12, 24)) + list(range(28, 32))

        self.price_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())

        self.vol_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())

        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.gated_layers = nn.Sequential(
            GatedResidualBlock(hidden_dim), GatedResidualBlock(hidden_dim)
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std

        x_price = x[:, :, self.price_indices]
        x_vol = x[:, :, self.vol_indices]

        p_emb = self.price_encoder(x_price)
        v_emb = self.vol_encoder(x_vol)

        combined = torch.cat([p_emb, v_emb], dim=-1)

        out, _ = self.rnn(combined)

        out = self.gated_layers(out)

        out = self.fc(out)
        return 6.0 * torch.tanh(out)
