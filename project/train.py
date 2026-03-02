import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_def import GenderAgeVGG16, build_eval_transforms, build_train_transforms, count_parameters


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(PROJECT_DIR / "FairFace")
DEFAULT_TRAIN_CSV = str(Path(DEFAULT_DATA_ROOT) / "fairface_label_train.csv")
DEFAULT_VAL_CSV = str(Path(DEFAULT_DATA_ROOT) / "fairface_label_val.csv")


AGE_BIN_TO_MIDPOINT = {
    "0-2": 1,
    "3-9": 6,
    "10-19": 15,
    "20-29": 25,
    "30-39": 35,
    "40-49": 45,
    "50-59": 55,
    "60-69": 65,
    "more than 70": 75,
    "70+": 75,
}


class FairFaceDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, transform=None, max_samples: Optional[int] = None, random_seed: int = 42):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

        if max_samples is not None and max_samples > 0 and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)

        required_cols = {"file", "gender", "age"}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    @staticmethod
    def age_to_value(age_raw) -> float:
        val = str(age_raw).strip().lower()
        if val in AGE_BIN_TO_MIDPOINT:
            return float(AGE_BIN_TO_MIDPOINT[val])
        if "-" in val:
            parts = val.split("-")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                return (float(parts[0]) + float(parts[1])) / 2.0
        if val.isdigit():
            return float(val)
        return 40.0

    @staticmethod
    def gender_to_value(gender_raw) -> float:
        val = str(gender_raw).strip().lower()
        if val == "male":
            return 1.0
        if val == "female":
            return 0.0
        return 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row["file"]).replace("\\", "/")
        img_path = self.data_root / rel_path

        if not img_path.exists():
            fallback = self.data_root / Path(rel_path).name
            img_path = fallback

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gender = torch.tensor([self.gender_to_value(row["gender"])], dtype=torch.float32)
        age = torch.tensor([self.age_to_value(row["age"])], dtype=torch.float32)
        return image, gender, age


def find_default_csv(data_root: str, split: str) -> str:
    root = Path(data_root)
    split = split.lower()
    candidates = list(root.rglob(f"*{split}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {data_root} for split={split}")
    candidates.sort()
    return str(candidates[0])


def resolve_effective_data_root(data_root: str, train_csv: Optional[str], val_csv: Optional[str]) -> str:
    if train_csv:
        return str(Path(train_csv).resolve().parent)
    if val_csv:
        return str(Path(val_csv).resolve().parent)

    train_csv_auto = find_default_csv(data_root, "train")
    return str(Path(train_csv_auto).resolve().parent)


def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    all_gender_true, all_gender_pred = [], []
    all_age_true, all_age_pred = [], []

    with torch.no_grad():
        for images, gender_targets, age_targets in loader:
            images = images.to(device)
            gender_targets = gender_targets.to(device)
            age_targets = age_targets.to(device)

            gender_out, age_out = model(images)

            gender_pred = (gender_out >= 0.5).float()
            all_gender_true.extend(gender_targets.cpu().numpy().reshape(-1).tolist())
            all_gender_pred.extend(gender_pred.cpu().numpy().reshape(-1).tolist())
            all_age_true.extend(age_targets.cpu().numpy().reshape(-1).tolist())
            all_age_pred.extend(age_out.cpu().numpy().reshape(-1).tolist())

    gender_acc = accuracy_score(all_gender_true, all_gender_pred)
    age_mae = mean_absolute_error(all_age_true, all_age_pred)
    score = gender_acc - (age_mae / 100.0)
    return gender_acc, age_mae, score


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = build_train_transforms(args.image_size) if args.use_augmentation else build_eval_transforms(args.image_size)
    val_transform = build_eval_transforms(args.image_size)

    train_csv = args.train_csv if args.train_csv else (DEFAULT_TRAIN_CSV if Path(DEFAULT_TRAIN_CSV).exists() else find_default_csv(args.data_root, "train"))
    val_csv = args.val_csv if args.val_csv else (DEFAULT_VAL_CSV if Path(DEFAULT_VAL_CSV).exists() else find_default_csv(args.data_root, "val"))
    effective_data_root = resolve_effective_data_root(args.data_root, train_csv, val_csv)

    print(f"Using train CSV: {train_csv}")
    print(f"Using val CSV: {val_csv}")
    print(f"Using effective data root: {effective_data_root}")

    train_ds = FairFaceDataset(train_csv, effective_data_root, train_transform, max_samples=args.train_limit)
    val_ds = FairFaceDataset(val_csv, effective_data_root, val_transform, max_samples=args.val_limit)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    use_pin_memory = device.type == "cuda"
    use_persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    model = GenderAgeVGG16(dropout=args.dropout, unfreeze_last_blocks=args.unfreeze_last_blocks).to(device)

    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    criterion_gender = nn.BCELoss()
    criterion_age = nn.L1Loss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    history = {
        "train_total_loss": [],
        "train_gender_loss": [],
        "train_age_loss": [],
        "val_gender_acc": [],
        "val_age_mae": [],
    }

    best_score = -1e9
    epochs_without_improvement = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "model.pth")

    for epoch in range(args.epochs):
        model.train()

        running_total, running_gender, running_age = 0.0, 0.0, 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, gender_targets, age_targets in progress:
            images = images.to(device)
            gender_targets = gender_targets.to(device)
            age_targets = age_targets.to(device)

            optimizer.zero_grad()
            gender_out, age_out = model(images)

            loss_gender = criterion_gender(gender_out, gender_targets)
            loss_age = criterion_age(age_out, age_targets)
            loss_total = args.gender_loss_weight * loss_gender + args.age_loss_weight * loss_age

            loss_total.backward()
            optimizer.step()

            running_total += loss_total.item()
            running_gender += loss_gender.item()
            running_age += loss_age.item()

            progress.set_postfix(
                total=f"{loss_total.item():.4f}",
                gender=f"{loss_gender.item():.4f}",
                age=f"{loss_age.item():.4f}",
            )

        train_total = running_total / len(train_loader)
        train_gender = running_gender / len(train_loader)
        train_age = running_age / len(train_loader)

        val_gender_acc, val_age_mae, val_score = evaluate(model, val_loader, device)

        history["train_total_loss"].append(train_total)
        history["train_gender_loss"].append(train_gender)
        history["train_age_loss"].append(train_age)
        history["val_gender_acc"].append(val_gender_acc)
        history["val_age_mae"].append(val_age_mae)

        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}: train_total={train_total:.4f}, train_gender={train_gender:.4f}, "
            f"train_age={train_age:.4f}, val_gender_acc={val_gender_acc:.4f}, val_age_mae={val_age_mae:.4f}, lr={current_lr:.6f}"
        )

        if val_score > (best_score + args.min_delta):
            best_score = val_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "unfreeze_last_blocks": args.unfreeze_last_blocks,
                    "params": {
                        "total": total_params,
                        "trainable": trainable_params,
                        "frozen": frozen_params,
                    },
                },
                best_model_path,
            )
            print(f"Saved best model to {best_model_path}")
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {epochs_without_improvement} epochs)")
            break

    plot_path = os.path.join(args.output_dir, "training_curves.png")
    num_epochs_plotted = len(history["train_total_loss"])
    if num_epochs_plotted > 0:
        epoch_axis = list(range(1, num_epochs_plotted + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_axis, history["train_total_loss"], marker="o", label="Train Total Loss")
        plt.plot(epoch_axis, history["train_gender_loss"], marker="o", label="Train Gender Loss")
        plt.plot(epoch_axis, history["train_age_loss"], marker="o", label="Train Age Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epoch_axis)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
    else:
        print("No completed epochs, skipping training curve plot.")

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Best composite score: {best_score:.4f}\n")
        if history["val_gender_acc"] and history["val_age_mae"]:
            f.write(f"Final val gender accuracy: {history['val_gender_acc'][-1]:.4f}\n")
            f.write(f"Final val age MAE: {history['val_age_mae'][-1]:.4f}\n")
        else:
            f.write("Final val gender accuracy: N/A\n")
            f.write("Final val age MAE: N/A\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Frozen parameters: {frozen_params}\n")

    print(f"Saved curves at {plot_path}")
    print(f"Saved metrics at {metrics_path}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="FairFace root folder",
    )
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1, help="Number of last VGG feature blocks to fine-tune")
    parser.add_argument("--gender-loss-weight", type=float, default=1.0)
    parser.add_argument("--age-loss-weight", type=float, default=0.15)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--lr-patience", type=int, default=2, help="Epochs to wait before reducing LR")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Epochs without val score improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val score improvement to reset patience")
    parser.add_argument("--train-limit", type=int, default=8000, help="Optional cap on number of training samples; use 0 for full set")
    parser.add_argument("--val-limit", type=int, default=2000, help="Optional cap on number of validation samples; use 0 for full set")
    parser.set_defaults(use_augmentation=True)
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false", help="Disable training augmentations")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
