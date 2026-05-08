import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from config import LEARNING_RATE, EPOCHS_GRID, PATIENCE, PLOT_DIR
from dataset import get_transforms, make_grid_loader
from models import CNNImproved


def q4_forward_pass(model, loader, device, steps=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    X_b, y_b = next(iter(loader))
    X_b, y_b = X_b.to(device), y_b.to(device)
    prev = None
    for s in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        arrow = " v" if prev and loss.item() < prev else ""
        print(f"  Step {s}: loss={loss.item():.6f}{arrow}")
        prev = loss.item()


def _one_epoch_train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (out.argmax(1) == y).sum().item()
        n += len(X)
    return total_loss / n, correct / n


@torch.no_grad()
def _one_epoch_eval(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(X)
        correct += (out.argmax(1) == y).sum().item()
        n += len(X)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, epochs, lr, weight_decay,
                device, verbose=True, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR smoothly decays lr from lr_max to ~0 over T_max steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_vl = float("inf")
    no_imp = 0
    best_state = None
    best_vacc = 0.0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _one_epoch_train(model, train_loader,
                                           criterion, optimizer, device)
        vl_loss, vl_acc = _one_epoch_eval(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_vl:
            best_vl = vl_loss
            best_vacc = vl_acc
            no_imp = 0
            # save snapshot of the best weights seen so far
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {ep:>2}/{epochs}  "
                  f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
                  f"tr_acc={tr_acc:.3f}  vl_acc={vl_acc:.3f}  "
                  f"lr={lr_now:.2e}")

        if no_imp >= patience:
            print(f"  Early stopping at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history, best_vacc


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    softmax = nn.Softmax(dim=1)
    for X, y in loader:
        logits = model(X.to(device))
        probs.append(softmax(logits).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds),
            np.concatenate(labels),
            np.concatenate(probs, 0))


def evaluate_classification(model, loader, class_names, device, split_name="val"):
    preds, labels, probs = collect_predictions(model, loader, device)
    top1 = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    pcf1 = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )[2]
    cm = confusion_matrix(labels, preds)
    print(f"\n[Q6] {split_name}: acc={top1:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    for i, idx in enumerate(np.argsort(pcf1)):
        tag = "  <- weakest" if i < 3 else ""
        print(f"  {class_names[idx]:<20} F1={pcf1[idx]:.4f}{tag}")
    print(classification_report(labels, preds,
                                target_names=class_names, zero_division=0))
    return preds, labels, probs, cm, pcf1


def grid_search(val_loader, class_names, device, num_classes):
    grid = {
        "filters": [32, 64],
        "lr": [1e-3, 3e-4],
        "batch_size": [32, 64],
        "weight_decay": [0, 1e-4],
    }
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    best_f1 = -1.0
    best_cfg = None
    best_mdl = None
    results = []

    print(f"\n[Q7] Grid: {len(combos)} combos x {EPOCHS_GRID} epochs ...")
    for combo in combos:
        cfg = dict(zip(keys, combo))
        loader = make_grid_loader(cfg["batch_size"])
        model = CNNImproved(num_classes, cfg["filters"]).to(device)
        train_model(model, loader, val_loader, EPOCHS_GRID,
                    cfg["lr"], cfg["weight_decay"], device,
                    verbose=False, patience=999)
        preds, labels, *_ = evaluate_classification(
            model, val_loader, class_names, device, "grid"
        )
        _, _, mf1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        tag = "  <- best" if mf1 > best_f1 else ""
        print(f"  {cfg}  F1={mf1:.4f}{tag}")
        results.append({**cfg, "macro_F1": mf1})
        if mf1 > best_f1:
            best_f1 = mf1
            best_cfg = cfg
            best_mdl = model

    print(f"\n[Q7] Best: {best_cfg}  F1={best_f1:.4f}")
    return best_mdl, best_cfg, pd.DataFrame(results).sort_values(
        "macro_F1", ascending=False
    )


def run_inference(model, folder, class_names, device, top_k=5):
    # top_k is clamped: requesting top-5 on a 2-class dataset would crash
    top_k = min(top_k, len(class_names))
    dataset = datasets.ImageFolder(folder, transform=get_transforms(False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    softmax = nn.Softmax(dim=1)
    rows = []
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        for X, _ in loader:
            probs = softmax(model(X.to(device))).cpu().numpy()
            for prob in probs:
                idx = np.argsort(prob)[::-1][:top_k]
                row = {"top1_pred": class_names[idx[0]]}
                for k in range(top_k):
                    row[f"top{k + 1}_class"] = class_names[idx[k]]
                    row[f"top{k + 1}_prob"] = round(prob[idx[k]], 4)
                rows.append(row)
    t1 = time.perf_counter()

    filenames = [os.path.basename(s[0]) for s in dataset.samples]
    df = pd.DataFrame(rows)
    df.insert(0, "filename", filenames[:len(df)])
    csv_path = os.path.join(PLOT_DIR, "inference_results.csv")
    df.to_csv(csv_path, index=False)

    dev_str = "GPU" if device.type == "cuda" else "CPU"
    throughput = len(dataset) / (t1 - t0)
    print(f"\n[Q9] {len(dataset)} images  {throughput:.1f} img/s ({dev_str})")
    print(f"  Saved: {csv_path}")
    return df
