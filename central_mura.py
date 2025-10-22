import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18


def freeze_all_but_fc(model):
    """Freezes all layers of the model except for the final fully connected layer."""
    print("🧊 Freezing all layers except the final classifier head.")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_last_block(model):
    """Unfreezes the final convolutional block (layer4) and the classifier head."""
    print("🔓 Unfreezing last block (layer4 + FC).")
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    """Unfreezes all layers of the model."""
    print("🔓 Unfreezing all layers for full model fine-tuning.")
    for param in model.parameters():
        param.requires_grad = True

def get_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler to handle class imbalance by oversampling
    the minority class, ensuring each batch has a balanced distribution.
    """
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def parse_label_from_path(path: str) -> int:
    """Return the TB label encoded in the filename."""
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    return int(stem.split('_')[-1])


# --- 2. Custom Dataset Definition ---
class ShenzhenDataset(Dataset):
    """PyTorch Dataset for the Shenzhen TB CXRs."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = self.dataframe['label'].astype(int).tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as exc:
            raise RuntimeError(f"Failed to open image at '{img_path}'.") from exc

        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. Model Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on the validation set and computes AUC."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels_float = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels_float)
            running_loss += loss.item() * inputs.size(0)
            preds_sigmoid = torch.sigmoid(outputs)
            preds_binary = preds_sigmoid > 0.5
            correct_predictions += (preds_binary == labels_float).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_sigmoid.cpu().numpy().flatten())

    val_loss = running_loss / total_samples
    val_acc = (correct_predictions / total_samples) * 100
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    return val_loss, val_acc, auc_score

# --- 4. Main Execution Block ---

if __name__ == '__main__':
    # --- Configuration ---
    QUICK_TEST = False
    QUICK_TEST_FRAC = 0.2
    NUM_EPOCHS = 6
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PARTITIONS_DIR = Path(__file__).resolve().parent / 'partitions'
    TRAIN_CSV = PARTITIONS_DIR / 'global_train.csv'
    VALID_CSV = PARTITIONS_DIR / 'global_valid.csv'
    print(f"Using device: {DEVICE}")

    # --- Data Preparation with AUGMENTATION ---
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- PHASE 1: CENTRALIZED BENCHMARK ---
    print("\n--- Starting Phase 1: Centralized Benchmark Training ---")

    if not TRAIN_CSV.exists() or not VALID_CSV.exists():
        raise FileNotFoundError(
            "Training/validation splits not found. Run partition.py before training."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    if QUICK_TEST:
        print(f"--- QUICK TEST MODE: Using {QUICK_TEST_FRAC * 100:.0f}% of the data ---")
        train_df = train_df.sample(frac=QUICK_TEST_FRAC, random_state=42)
        valid_df = valid_df.sample(frac=QUICK_TEST_FRAC, random_state=42)

    train_dataset = ShenzhenDataset(train_df, transform=train_transforms)
    valid_dataset = ShenzhenDataset(valid_df, transform=valid_transforms)

    print(
        f"Training samples: {len(train_dataset)} (TB prevalence: {np.mean(train_dataset.labels) * 100:.2f}%)"
    )
    print(
        f"Validation samples: {len(valid_dataset)} (TB prevalence: {np.mean(valid_dataset.labels) * 100:.2f}%)"
    )

    # --- Handling Class Imbalance with WeightedRandomSampler ---
    print("Setting up WeightedRandomSampler to handle class imbalance...")
    train_sampler = get_weighted_sampler(train_dataset.labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Setup with Dropout ---
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop with Gradual Unfreezing ---
    best_val_auc = 0.0
    best_model_wts = None

    # Initial phase: Train only the classifier head
    freeze_all_but_fc(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        # Granular Unfreezing Schedule
        if epoch == 2:
            unfreeze_last_block(model)
            # Re-initialize optimizer for the newly trainable parameters with a smaller LR
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        if epoch == 4:
            unfreeze_all(model)
            # Re-initialize optimizer for the full model with a very small LR
            optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auc = evaluate_model(model, valid_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} -> "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved with Val AUC: {best_val_auc:.4f}")

    print("\nCentralized training complete.")
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model with Final Validation AUC: {best_val_auc:.4f}")
        models_dir = Path(__file__).resolve().parent / 'models'
        models_dir.mkdir(exist_ok=True)
        checkpoint_path = models_dir / 'best_resnet18_shenzhen.pth'
        torch.save(best_model_wts, checkpoint_path)
        print(f"✅ Best model weights saved to {checkpoint_path}")

