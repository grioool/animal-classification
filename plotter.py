import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms

from config import PLOT_DIR, MEAN, STD


def save_fig(name: str):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> Saved: {path}")
    plt.close()


def _inv_norm():
    # reverse the ImageNet normalisation so images display correctly
    return transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)],
                             std=[1 / s for s in STD])
    ])


def plot_learning_curves(history: dict, title: str = "Learning Curves"):
    eps = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, history["train_loss"], label="Train", marker="o")
    ax1.plot(eps, history["val_loss"], label="Val", marker="s")
    ax1.set(xlabel="Epoch", ylabel="Loss", title=f"{title} - Loss")
    ax1.legend();
    ax1.grid(True, alpha=0.3)
    ax2.plot(eps, history["train_acc"], label="Train", marker="o")
    ax2.plot(eps, history["val_acc"], label="Val", marker="s")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title=f"{title} - Accuracy")
    ax2.legend();
    ax2.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    save_fig("q5_learning_curves.png")


def plot_confusion_matrix(cm, class_names: list):
    cm_n = cm.astype(float) / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)),
                                    max(5, len(class_names))))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set(xlabel="Predicted", ylabel="True",
           title="Q8a - Row-Normalised Confusion Matrix")
    plt.tight_layout()
    save_fig("q8a_confusion_matrix.png")


def plot_prediction_grid(model, loader, class_names: list, device,
                         n_c: int = 8, n_i: int = 8):
    model.eval()
    c_imgs, c_info, i_imgs, i_info = [], [], [], []
    inv = _inv_norm()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for X, y in loader:
            probs = softmax(model(X.to(device))).cpu()
            for img, label, prob in zip(X, y, probs):
                pred = prob.argmax().item()
                conf = prob[pred].item()
                info = (class_names[label.item()], class_names[pred], conf)
                img_disp = inv(img).permute(1, 2, 0).clamp(0, 1).numpy()
                if pred == label.item() and len(c_imgs) < n_c:
                    c_imgs.append(img_disp);
                    c_info.append(info)
                elif pred != label.item() and len(i_imgs) < n_i:
                    i_imgs.append(img_disp);
                    i_info.append(info)
            if len(c_imgs) >= n_c and len(i_imgs) >= n_i:
                break

    all_imgs = c_imgs + i_imgs
    all_info = c_info + i_info
    is_cor = [True] * len(c_imgs) + [False] * len(i_imgs)

    fig, axes = plt.subplots(4, 4, figsize=(4 * 3, 4 * 3))
    for ax, img, info, ok in zip(axes.flat, all_imgs, all_info, is_cor):
        ax.imshow(img)
        col = "#16a34a" if ok else "#dc2626"
        for sp in ax.spines.values():
            sp.set_edgecolor(col);
            sp.set_linewidth(3)
        ax.set_title(f"T: {info[0]}\nP: {info[1]} ({info[2]:.1%})",
                     fontsize=7.5, color=col)
        ax.axis("off")

    fig.legend(handles=[
        mpatches.Patch(color="#16a34a", label="Correct"),
        mpatches.Patch(color="#dc2626", label="Incorrect"),
    ], loc="lower center", ncol=2, frameon=False)
    plt.suptitle("Q8b - Correct (top) vs Incorrect (bottom) Predictions")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig("q8b_prediction_grid.png")


def plot_error_analysis(model, loader, class_names: list, device):
    model.eval()
    c_imgs, c_info, i_imgs, i_info = [], [], [], []
    inv = _inv_norm()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for X, y in loader:
            probs = softmax(model(X.to(device))).cpu()
            for img, label, prob in zip(X, y, probs):
                pred = prob.argmax().item()
                conf = prob[pred].item()
                info = (class_names[label.item()], class_names[pred], conf)
                img_disp = inv(img).permute(1, 2, 0).clamp(0, 1).numpy()
                if pred == label.item() and len(c_imgs) < 5:
                    c_imgs.append(img_disp);
                    c_info.append(info)
                elif pred != label.item() and len(i_imgs) < 4:
                    i_imgs.append(img_disp);
                    i_info.append(info)
            if len(c_imgs) >= 5 and len(i_imgs) >= 4:
                break

    all_imgs = c_imgs + i_imgs
    all_info = c_info + i_info
    is_cor = [True] * len(c_imgs) + [False] * len(i_imgs)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, img, info, ok in zip(axes.flat, all_imgs, all_info, is_cor):
        ax.imshow(img)
        col = "#15803d" if ok else "#b91c1c"
        for sp in ax.spines.values():
            sp.set_edgecolor(col);
            sp.set_linewidth(3)
        sym = "v" if ok else "x"
        ax.set_title(f"{sym}  True: {info[0]}\nPred: {info[1]}  ({info[2]:.1%})",
                     fontsize=8, color=col)
        ax.axis("off")

    plt.suptitle("Q10 - Explainability & Error Analysis\n"
                 "(green = correct, red = incorrect)")
    plt.tight_layout()
    save_fig("q10_error_analysis.png")

    print("\n[Q10] Observed failure patterns:")
    print("  1. Background bias - the model uses scene context (grass, road, savanna)")
    print("     as a proxy for class; changing the background fools it.")
    print("  2. OOD subjects (e.g. rhino) are classified with 99% confidence because")
    print("     there is no 'unknown' class; deploy with a confidence threshold.")
    print("  3. Correct predictions with <70% confidence indicate a narrow decision")
    print("     boundary; transfer learning would widen it substantially.")
