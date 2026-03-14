import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import MultiTaskModel
from src.data_loader import FreshnessDataset, get_transforms


def train():
    print("🚀 Starting training...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Using device: {device}")

    # Dataset
    train_ds = FreshnessDataset(
        csv_file="data/annotations.csv",
        root_dir="data",
        transform=get_transforms()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    # Model
    model = MultiTaskModel().to(device)

    # Losses
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")

        for i, (x, y_days, y_cls) in enumerate(train_loader):
            x = x.to(device)
            y_days = y_days.to(device)
            y_cls = y_cls.to(device)

            optimizer.zero_grad()

            pred_days, pred_cls = model(x)

            loss_days = reg_loss_fn(pred_days.squeeze(), y_days)
            loss_cls = cls_loss_fn(pred_cls, y_cls)

            loss = loss_days + loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"  Batch {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    print("\n🎉 Training complete!")
    print("💾 Model saved to checkpoints/best_model.pt")


if __name__ == "__main__":
    train()
