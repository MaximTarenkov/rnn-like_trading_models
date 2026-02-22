import os
import random
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
import numpy as np

from model import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def weighted_pearson_loss(y_pred, y_true):
    y_pred = y_pred[:, 99:, :]
    y_true = y_true[:, 99:, :]

    pred_flat = y_pred.reshape(-1, 2)
    true_flat = y_true.reshape(-1, 2)

    global_loss = 0.0
    for i in range(2):
        p = pred_flat[:, i]
        t = true_flat[:, i]
        w = torch.abs(t).clamp(min=1e-8)
        sw = torch.sum(w)

        sn_p = p - torch.sum(p * w) / sw
        sn_t = t - torch.sum(t * w) / sw

        cov = torch.sum(w * sn_p * sn_t) / sw
        var_p = torch.sum(w * sn_p**2) / sw
        var_t = torch.sum(w * sn_t**2) / sw

        global_loss -= cov / (
            torch.sqrt(var_p + 1e-8) * torch.sqrt(var_t + 1e-8) + 1e-8
        )
    global_loss /= 2.0

    local_loss = 0.0
    for i in range(2):
        p = y_pred[:, :, i]
        t = y_true[:, :, i]
        w = torch.abs(t).clamp(min=1e-8)
        sw = torch.sum(w, dim=1, keepdim=True)

        sn_p = p - torch.sum(p * w, dim=1, keepdim=True) / sw
        sn_t = t - torch.sum(t * w, dim=1, keepdim=True) / sw

        cov = torch.sum(w * sn_p * sn_t, dim=1, keepdim=True) / sw
        var_p = torch.sum(w * sn_p**2, dim=1, keepdim=True) / sw
        var_t = torch.sum(w * sn_t**2, dim=1, keepdim=True) / sw

        local_loss -= torch.mean(
            cov / (torch.sqrt(var_p + 1e-8) * torch.sqrt(var_t + 1e-8) + 1e-8)
        )
    local_loss /= 2.0

    return 0.7 * global_loss + 0.3 * local_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = weighted_pearson_loss(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_global(model, loader, device):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)

            all_preds.append(y_pred.cpu())
            all_trues.append(y_batch.cpu())

    y_pred_all = torch.cat(all_preds, dim=0)  # [N, 1000, 2]
    y_true_all = torch.cat(all_trues, dim=0)  # [N, 1000, 2]

    y_pred_all = y_pred_all[:, 99:, :]
    y_true_all = y_true_all[:, 99:, :]

    global_loss = 0.0

    for i in range(2):
        pred = y_pred_all[:, :, i].flatten()
        true = y_true_all[:, :, i].flatten()

        weights = torch.abs(true).clamp(min=1e-8)
        sum_w = torch.sum(weights)

        mean_true = torch.sum(true * weights) / sum_w
        mean_pred = torch.sum(pred * weights) / sum_w

        dev_true = true - mean_true
        dev_pred = pred - mean_pred

        cov = torch.sum(weights * dev_true * dev_pred) / sum_w
        var_true = torch.sum(weights * dev_true**2) / sum_w
        var_pred = torch.sum(weights * dev_pred**2) / sum_w

        corr = cov / (torch.sqrt(var_true + 1e-8) * torch.sqrt(var_pred + 1e-8) + 1e-8)
        global_loss -= corr.item()

    return global_loss / 2.0


def run_training(seed, train_loader, valid_loader, device):
    set_seed(seed)

    model = TradingModel().to(device)
    swa_model = AveragedModel(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 100
    patience = 2
    lr_factor = 0.5
    min_lr = 1e-4
    swa_start_epoch = 3

    best_valid_loss = float("inf")

    best_state = {
        "model": copy.deepcopy(model.state_dict()),
        "swa": copy.deepcopy(swa_model.state_dict()),
    }

    wait = 0

    log_file = f"training_log_seed_{seed}.txt"
    model_file = f"best_model_seed_{seed}.pth"
    swa_file = f"best_swa_seed_{seed}.pth"

    with open(log_file, "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write("Epoch | Train Loss | Train Corr | Valid Corr | LR\n")
        f.write("-" * 60 + "\n")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        valid_loss = evaluate_global(model, valid_loader, device)

        train_corr = -train_loss
        valid_corr = -valid_loss
        current_lr = optimizer.param_groups[0]["lr"]

        log_str = f"Epoch {epoch+1:02d} | Train Corr: {train_corr:.4f} | Valid Corr: {valid_corr:.4f} | LR: {current_lr:.6f}"
        print(log_str)

        with open(log_file, "a") as f:
            f.write(log_str + "\n")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state["model"] = copy.deepcopy(model.state_dict())
            best_state["swa"] = copy.deepcopy(swa_model.state_dict())

            torch.save(best_state["model"], model_file)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                new_lr = current_lr * lr_factor

                if new_lr >= min_lr:
                    msg = f"--- ПЛАТО: Откат к лучшим весам (Model + SWA) и снижение LR: {current_lr:.6f} -> {new_lr:.6f} ---"
                    print(msg)
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")

                    model.load_state_dict(best_state["model"])
                    swa_model.load_state_dict(best_state["swa"])

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    wait = 0
                else:
                    msg = "--- Достигнут минимальный LR. Остановка обучения ---"
                    print(msg)
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")
                    break

    print("\nEvaluating Final SWA Model...")
    swa_valid_loss = evaluate_global(swa_model, valid_loader, device)
    swa_valid_corr = -swa_valid_loss

    print(f"SWA Valid Corr: {swa_valid_corr:.6f}")

    final_best_corr = max(-best_valid_loss, swa_valid_corr)

    with open(log_file, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Final Best Standard Corr: {-best_valid_loss:.6f}\n")
        f.write(f"Final SWA Corr: {swa_valid_corr:.6f}\n")

    torch.save(swa_model.state_dict(), swa_file)

    return final_best_corr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets...")
    train_dataset = FinSeriesDataset("datasets/train.parquet")
    valid_dataset = FinSeriesDataset("datasets/valid.parquet")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    seeds = [random.randint(0, 1000000) for _ in range(150)]

    csv_file = "seed_results.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Max_Valid_Corr"])

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Starting training for SEED: {seed}")
        print(f"{'='*40}")

        max_corr = run_training(seed, train_loader, valid_loader, device)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seed, f"{max_corr:.6f}"])

        print(f"Finished SEED {seed}. Best Result: {max_corr:.6f}")


if __name__ == "__main__":
    main()
