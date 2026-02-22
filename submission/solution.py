import os
import numpy as np
import torch
import torch.nn as nn


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


class TradingModel_Gated(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.price_indices = list(range(0, 12)) + list(range(24, 28))
        self.vol_indices = list(range(12, 24)) + list(range(28, 32))
        self.price_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.vol_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.rnn = nn.GRU(128, hidden_dim, num_layers, batch_first=True)
        self.gated_layers = nn.Sequential(
            GatedResidualBlock(hidden_dim), GatedResidualBlock(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hx=None):
        x_price = x[:, :, self.price_indices]
        x_vol = x[:, :, self.vol_indices]
        p_emb = self.price_encoder(x_price)
        v_emb = self.vol_encoder(x_vol)
        combined = torch.cat([p_emb, v_emb], dim=-1)
        out, hx = self.rnn(combined, hx)
        out = self.gated_layers(out)
        out = self.fc(out)
        return 6.0 * torch.tanh(out), hx


class TradingModel_Simple(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.price_indices = list(range(0, 12)) + list(range(24, 28))
        self.vol_indices = list(range(12, 24)) + list(range(28, 32))
        self.price_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.vol_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.rnn = nn.GRU(128, hidden_dim, num_layers, batch_first=True)
        self.glu_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hx=None):
        x_price = x[:, :, self.price_indices]
        x_vol = x[:, :, self.vol_indices]
        p_emb = self.price_encoder(x_price)
        v_emb = self.vol_encoder(x_vol)
        combined = torch.cat([p_emb, v_emb], dim=-1)
        out, hx = self.rnn(combined, hx)
        out = self.glu_proj(out)
        out = self.glu(out)
        out = self.fc(out)
        return 6.0 * torch.tanh(out), hx


class PredictionModel:
    def __init__(self, t0_paths=["model_t0_0.pth", "model_t0_1.pth"], t1_paths=["model_t1_0.pth", "model_t1_1.pth"]):
        self.device = torch.device("cpu")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_t0 = self._load_models(t0_paths)
        self.models_t1 = self._load_models(t1_paths)
        self.current_seq_ix = None
        self.h_t0 = [None] * len(self.models_t0)
        self.h_t1 = [None] * len(self.models_t1)

    def _load_models(self, paths):
        loaded_models = []
        for m_path in paths:
            full_path = os.path.join(self.base_dir, m_path)
            if not os.path.exists(full_path):
                print(f"Warning: Model file {full_path} not found.")
                continue
            raw_state = torch.load(full_path, map_location=self.device)
            if isinstance(raw_state, dict):
                if "swa" in raw_state:
                    state_dict = raw_state["swa"]
                elif "model" in raw_state:
                    state_dict = raw_state["model"]
                elif "state_dict" in raw_state:
                    state_dict = raw_state["state_dict"]
                else:
                    state_dict = raw_state
            else:
                state_dict = raw_state
            clean_sd = {}
            for k, v in state_dict.items():
                if k == "n_averaged":
                    continue
                new_k = k.replace("module.", "")
                while new_k.startswith("module."):
                    new_k = new_k.replace("module.", "")
                clean_sd[new_k] = v
            keys_str = " ".join(clean_sd.keys())
            if "gated_layers" in keys_str:
                model = TradingModel_Gated().to(self.device)
            else:
                model = TradingModel_Simple().to(self.device)
            try:
                model.load_state_dict(clean_sd, strict=True)
            except RuntimeError:
                model.load_state_dict(clean_sd, strict=False)
            model.eval()
            loaded_models.append(model)
        return loaded_models

    def predict(self, data_point) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.h_t0 = [None] * len(self.models_t0)
            self.h_t1 = [None] * len(self.models_t1)
        x = (
            torch.tensor(data_point.state, dtype=torch.float32)
            .view(1, 1, -1)
            .to(self.device)
        )
        sum_t0 = 0.0
        if len(self.models_t0) > 0:
            with torch.no_grad():
                for i, model in enumerate(self.models_t0):
                    out, self.h_t0[i] = model(x, self.h_t0[i])
                    sum_t0 += out[:, :, 0]
            avg_t0 = sum_t0 / len(self.models_t0)
        else:
            avg_t0 = torch.zeros(1, 1).to(self.device)
        sum_t1 = 0.0
        if len(self.models_t1) > 0:
            with torch.no_grad():
                for i, model in enumerate(self.models_t1):
                    out, self.h_t1[i] = model(x, self.h_t1[i])
                    sum_t1 += out[:, :, 1]
            avg_t1 = sum_t1 / len(self.models_t1)
        else:
            avg_t1 = torch.zeros(1, 1).to(self.device)
        if not data_point.need_prediction:
            return None
        result = torch.cat([avg_t0, avg_t1], dim=-1)
        return result[0, :].cpu().numpy()
