import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FinSeriesDataset(Dataset):
    def __init__(self, parquet_path):
        super().__init__()
        df = pd.read_parquet(parquet_path)
        df['step_in_seq'] = df['step_in_seq'].astype(int)
        df = df.sort_values(['seq_ix', 'step_in_seq'])
        
        feature_cols = [f'p{i}' for i in range(12)] + \
                       [f'v{i}' for i in range(12)] + \
                       [f'dp{i}' for i in range(4)] + \
                       [f'dv{i}' for i in range(4)]
        target_cols = ['t0', 't1']
        
        num_seqs = len(df) // 1000
        
        x_data = df[feature_cols].values.astype(np.float32)
        y_data = df[target_cols].values.astype(np.float32)
        
        self.x = torch.tensor(x_data).view(num_seqs, 1000, 32)
        self.y = torch.tensor(y_data).view(num_seqs, 1000, 2)
        
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TradingModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return 6.0 * torch.tanh(out)

def weighted_pearson_loss(y_pred, y_true):
    y_pred = y_pred[:, 99:, :]
    y_true = y_true[:, 99:, :]
    
    loss = 0.0
    for target_idx in range(2):
        pred = y_pred[:, :, target_idx].flatten()
        true = y_true[:, :, target_idx].flatten()
        
        weights = torch.abs(true)
        weights = torch.maximum(weights, torch.tensor(1e-8, device=weights.device))
        
        sum_w = torch.sum(weights)
        
        mean_true = torch.sum(true * weights) / sum_w
        mean_pred = torch.sum(pred * weights) / sum_w
        
        dev_true = true - mean_true
        dev_pred = pred - mean_pred
        
        cov = torch.sum(weights * dev_true * dev_pred) / sum_w
        var_true = torch.sum(weights * dev_true**2) / sum_w
        var_pred = torch.sum(weights * dev_pred**2) / sum_w
        
        corr = cov / (torch.sqrt(var_true + 1e-8) * torch.sqrt(var_pred + 1e-8) + 1e-8)
        loss -= corr
        
    return loss / 2.0

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = FinSeriesDataset('datasets/train.parquet')
    valid_dataset = FinSeriesDataset('datasets/valid.parquet')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    model = TradingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            loss = weighted_pearson_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(x_batch)
                loss = weighted_pearson_loss(y_pred, y_batch)
                valid_loss += loss.item()
                
        valid_loss /= len(valid_loader)
        
        train_corr = -train_loss
        valid_corr = -valid_loss
        
        print(f"Epoch {epoch+1:02d} | Train Corr: {train_corr:.4f} | Valid Corr: {valid_corr:.4f}")

if __name__ == '__main__':
    train_model()