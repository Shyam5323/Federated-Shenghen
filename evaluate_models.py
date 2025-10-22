import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class ShenzhenDataset(Dataset):
    """Simple dataset that reads absolute image paths and binary labels."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = self.dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = Path(row["path"])
        label = int(row["label"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV at {csv_path}.")
    df = pd.read_csv(csv_path)
    required_cols = {"path", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")
    return df


def instantiate_model(weights_path: Path, device: torch.device) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_outputs(model: nn.Module, dataloader: DataLoader, device: torch.device):
    logits = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            logits.extend(outputs.squeeze(1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    logits = np.array(logits)
    labels = np.array(labels)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return labels, probs


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    if len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    metrics["report"] = classification_report(
        labels, preds, target_names=["Negative", "Positive"], zero_division=0
    )

    return metrics


def summarize_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ["threshold", "accuracy", "precision", "recall", "f1", "auc"]
        if key in metrics
    }


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_metrics.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)
    print(f"✅ Metrics written to {json_path}")


def plot_metric_comparison(results: dict, output_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    central_values = [results["centralized"].get(m, float("nan")) for m in metrics]
    federated_values = [results["federated"].get(m, float("nan")) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, central_values, width, label="Centralized")
    ax.bar(x + width / 2, federated_values, width, label="Federated")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Centralized vs Federated Performance")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plot_path = output_dir / "metric_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"📊 Metric comparison chart saved to {plot_path}")


def plot_roc_curves(curve_data: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    plotted = False
    for name, (y_true, y_score) in curve_data.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
        plotted = True

    if not plotted:
        plt.close(fig)
        print("⚠️ Skipped ROC plot (insufficient class variety).")
        return

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plot_path = output_dir / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"📈 ROC curve saved to {plot_path}")


def plot_confusion(matrix, title: str, output_dir: Path) -> None:
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    plot_path = output_dir / f"confusion_{title.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"📌 Confusion matrix saved to {plot_path}")


def plot_threshold_sweep(
    thresholds: np.ndarray,
    central_data: list,
    federated_data: list,
    output_dir: Path,
) -> None:
    if len(thresholds) == 0:
        return

    metrics_to_plot = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(len(metrics_to_plot), 1, sharex=True, figsize=(8, 10))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        central_values = [item.get(metric, float("nan")) for item in central_data]
        federated_values = [item.get(metric, float("nan")) for item in federated_data]

        ax.plot(thresholds, central_values, marker="o", label="Centralized")
        ax.plot(thresholds, federated_values, marker="s", label="Federated")
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Decision Threshold")
    fig.suptitle("Threshold Sweep (Centralized vs Federated)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    plot_path = output_dir / "threshold_sweep.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"📉 Threshold sweep plot saved to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate centralized and federated models on the Shenzhen test set.")
    parser.add_argument("--strategy", default="iid", choices=["iid", "label_skew", "pathological_non_iid"], help="Federated partition strategy to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--output_dir", default="analysis", help="Directory to store comparison results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default decision threshold applied to both models")
    parser.add_argument("--central-threshold", type=float, default=None, help="Override threshold for centralized model")
    parser.add_argument("--federated-threshold", type=float, default=None, help="Override threshold for federated model")
    parser.add_argument("--threshold-min", type=float, default=None, help="Lower bound for threshold sweep (inclusive)")
    parser.add_argument("--threshold-max", type=float, default=None, help="Upper bound for threshold sweep (inclusive)")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Step size for threshold sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partitions_dir = Path(__file__).resolve().parent / "partitions"
    test_csv = partitions_dir / "global_test.csv"
    test_df = load_dataframe(test_csv)

    transforms_pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ShenzhenDataset(test_df, transform=transforms_pipeline)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    models_dir = Path(__file__).resolve().parent / "models"
    central_path = models_dir / "best_resnet18_shenzhen.pth"
    federated_path = models_dir / f"best_federated_shenzhen_{args.strategy}.pth"

    central_model = instantiate_model(central_path, device)
    federated_model = instantiate_model(federated_path, device)

    central_labels, central_probs = collect_outputs(central_model, dataloader, device)
    federated_labels, federated_probs = collect_outputs(federated_model, dataloader, device)

    central_threshold = args.central_threshold if args.central_threshold is not None else args.threshold
    federated_threshold = args.federated_threshold if args.federated_threshold is not None else args.threshold

    central_metrics = compute_metrics(central_labels, central_probs, central_threshold)
    federated_metrics = compute_metrics(federated_labels, federated_probs, federated_threshold)

    results = {
        "device": str(device),
        "test_samples": len(test_dataset),
        "strategy": args.strategy,
        "thresholds": {
            "centralized": central_threshold,
            "federated": federated_threshold,
        },
        "centralized": central_metrics,
        "federated": federated_metrics,
    }

    output_dir = Path(args.output_dir)

    sweep_thresholds = None
    if args.threshold_min is not None:
        if args.threshold_max is None:
            raise ValueError("--threshold-max must be provided when --threshold-min is set")
        if args.threshold_step <= 0:
            raise ValueError("--threshold-step must be positive")
        if args.threshold_min > args.threshold_max:
            raise ValueError("--threshold-min cannot exceed --threshold-max")

        sweep_thresholds = np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        )
        sweep_thresholds = np.clip(sweep_thresholds, 0.0, 1.0)

        central_summaries = [
            summarize_metrics(compute_metrics(central_labels, central_probs, thr))
            for thr in sweep_thresholds
        ]
        federated_summaries = [
            summarize_metrics(compute_metrics(federated_labels, federated_probs, thr))
            for thr in sweep_thresholds
        ]

        results["threshold_sweep"] = {
            "thresholds": sweep_thresholds.tolist(),
            "centralized": central_summaries,
            "federated": federated_summaries,
        }

    save_results(results, output_dir)

    plot_metric_comparison(results, output_dir)
    plot_roc_curves(
        {
            "Centralized": (central_labels, central_probs),
            "Federated": (federated_labels, federated_probs),
        },
        output_dir,
    )

    plot_confusion(central_metrics["confusion_matrix"], "Centralized", output_dir)
    plot_confusion(federated_metrics["confusion_matrix"], "Federated", output_dir)

    if sweep_thresholds is not None:
        plot_threshold_sweep(sweep_thresholds, central_summaries, federated_summaries, output_dir)


if __name__ == "__main__":
    main()
