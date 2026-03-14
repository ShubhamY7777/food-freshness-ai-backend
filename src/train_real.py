import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.model import MultiTaskModel
from src.data_loader import FreshnessDataset, get_transforms


def train():

    print("🚀 Starting Training...\n")

    # -------------------------------------------------
    # Absolute Path Setup (NO MORE PATH ERRORS)
    # -------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, "dataset", "annotations_final.csv")
    ROOT_DIR = os.path.join(BASE_DIR, "dataset")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------
    # Config
    # -------------------------------------------------
    NUM_CLASSES = 23   # 🔥 Change to 19 after fixing duplicate folders
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")
    print(f"📄 Using CSV: {CSV_PATH}\n")

    # -------------------------------------------------
    # Load CSV
    # -------------------------------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        stratify=df["label_state"],
        random_state=42
    )

    # -------------------------------------------------
    # Dataset (Using DataFrame directly)
    # -------------------------------------------------
    train_dataset = FreshnessDataset(
        dataframe=train_df,
        root_dir=ROOT_DIR,
        transform=get_transforms()
    )

    val_dataset = FreshnessDataset(
        dataframe=val_df,
        root_dir=ROOT_DIR,
        transform=get_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = MultiTaskModel(num_classes=NUM_CLASSES).to(device)

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    for epoch in range(EPOCHS):

        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, days, labels in tqdm(train_loader):

            images = images.to(device)
            days = days.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred_days, pred_cls = model(images)

            loss_days = reg_loss_fn(pred_days, days)
            loss_cls = cls_loss_fn(pred_cls, labels)

            loss = loss_days + loss_cls
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(pred_cls, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, days, labels in tqdm(val_loader):

                images = images.to(device)
                days = days.to(device)
                labels = labels.to(device)

                pred_days, pred_cls = model(images)

                loss_days = reg_loss_fn(pred_days, days)
                loss_cls = cls_loss_fn(pred_cls, labels)

                loss = loss_days + loss_cls
                val_loss += loss.item()

                _, predicted = torch.max(pred_cls, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # ---------------- Save Best ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print("💾 Best model saved!")

    print("\n🎉 Training Complete!")


if __name__ == "__main__":
    train()