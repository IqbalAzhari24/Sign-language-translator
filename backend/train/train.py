import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from app.model import SignSequenceClassifier
from train.dataset import SignDataset

SPLIT_SEED = 0


def train(
    dataset_root,
    hands_detector,
    checkpoint_path,
    epochs=20,
    sequence_length=30,
    batch_size=8,
    lr=1e-3,
):
    dataset = SignDataset(dataset_root, hands_detector, sequence_length=sequence_length)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, _val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = SignSequenceClassifier(num_classes=len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

    torch.save(
        {"model_state": model.state_dict(), "classes": dataset.classes},
        checkpoint_path,
    )
    return losses


if __name__ == "__main__":
    from app.landmarks import create_hand_landmarker

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path laid out as <root>/<class>/<sample>/*.jpg")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    detector = create_hand_landmarker()
    try:
        train(
            args.dataset_root,
            detector,
            args.checkpoint_path,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
        )
    finally:
        detector.close()
