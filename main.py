import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import (
    SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS_MAIN, DATA_TEST,
)
from dataset import build_loaders
from models import BaselineLinear, CNNv1, CNNv2, CNNImproved, build_model
from trainer import (
    q4_forward_pass, train_model, evaluate_classification,
    grid_search, run_inference,
)
from plotter import (
    plot_learning_curves, plot_confusion_matrix,
    plot_prediction_grid, plot_error_analysis,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n{'-' * 60}")

    train_loader, val_loader, test_loader, class_names = build_loaders(
        BATCH_SIZE, augment=True
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    print("\n" + "-" * 60 + "\nQ1 - Baseline Linear")
    baseline = BaselineLinear(num_classes).to(device)
    train_model(baseline, train_loader, val_loader,
                EPOCHS_MAIN, LEARNING_RATE, WEIGHT_DECAY, device)
    bp, bl, *_ = evaluate_classification(baseline, val_loader,
                                         class_names, device, "Q1-val")
    print(f"  Baseline val acc: {(bp == bl).mean():.4f}")

    print("\n" + "-" * 60 + "\nQ2 - CNNv1 (no activation)");
    print(CNNv1(num_classes))
    print("\nQ3 - CNNv2 (+ ReLU)");
    print(CNNv2(num_classes))

    print("\n" + "-" * 60 + "\nQ4 - Forward pass check")
    q4_forward_pass(CNNImproved(num_classes).to(device), train_loader, device)

    print("\n" + "-" * 60 + f"\nQ5 - Training (up to {EPOCHS_MAIN} epochs)")
    model_main = build_model(num_classes).to(device)
    print(model_main)
    history, best_vacc = train_model(
        model_main, train_loader, val_loader,
        EPOCHS_MAIN, LEARNING_RATE, WEIGHT_DECAY, device
    )
    print(f"\n  Best val accuracy: {best_vacc:.4f}")
    plot_learning_curves(history, "Q5 - Improved CNN Learning Curves")

    print("\n" + "-" * 60 + "\nQ6 - Classification metrics")
    preds, labels, probs, cm, _ = evaluate_classification(
        model_main, val_loader, class_names, device, "Q6-val"
    )

    print("\n" + "-" * 60 + "\nQ7 - Grid search")
    best_model, best_cfg, grid_df = grid_search(
        val_loader, class_names, device, num_classes
    )
    print(grid_df.to_string(index=False))

    print("\n" + "-" * 60 + "\nQ8 - Visual diagnostics")
    _, _, _, cm2, _ = evaluate_classification(
        best_model, val_loader, class_names, device, "Q8-val"
    )
    plot_confusion_matrix(cm2, class_names)
    plot_prediction_grid(best_model, val_loader, class_names, device)

    print("\n" + "-" * 60 + "\nQ9 - Inference CSV")
    run_inference(best_model, DATA_TEST, class_names, device, top_k=5)

    print("\n" + "-" * 60 + "\nQ10 - Error analysis")
    plot_error_analysis(best_model, val_loader, class_names, device)


if __name__ == "__main__":
    main()
