import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "oa": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=[0, 1]),
        "confusion": matrix,
    }


def _forward(model, x1, x2):
    return model(torch.cat([x1, x2], dim=1))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    labels = []
    predictions = []
    for x1, x2, target in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        target = torch.clamp(target.to(device, non_blocking=True), 0, 1).long()
        output = _forward(model, x1, x2)
        loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        labels.extend(target.detach().cpu().numpy())
        predictions.extend(output.detach().argmax(dim=1).cpu().numpy())
    metrics = calculate_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []
    for x1, x2, target in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        target = torch.clamp(target.to(device, non_blocking=True), 0, 1).long()
        output = _forward(model, x1, x2)
        total_loss += criterion(output, target).item()
        labels.extend(target.cpu().numpy())
        predictions.extend(output.argmax(dim=1).cpu().numpy())
    metrics = calculate_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def _score(metrics):
    return (
        metrics["oa"]
        + metrics["f1"]
        + metrics["precision"]
        + metrics["recall"]
        + metrics["kappa"]
    )


def run_training(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    criterion,
    device,
    scheduler,
    best_model_path,
    validate_every=1,
    patience=15,
):
    start_time = time.time()
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_metrics = None
        if (epoch + 1) % validate_every == 0 or epoch + 1 == epochs:
            test_metrics = evaluate(model, test_loader, criterion, device)
            score = _score(test_metrics)
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                stale_epochs = 0
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "score": float(best_score),
                        "metrics": {
                            key: float(value)
                            for key, value in test_metrics.items()
                            if key != "confusion"
                        },
                    },
                    best_model_path,
                )
            else:
                stale_epochs += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "test": test_metrics,
            }
        )
        test_loss = "n/a" if test_metrics is None else f"{test_metrics['loss']:.4f}"
        test_oa = "n/a" if test_metrics is None else f"{test_metrics['oa']:.4f}"
        test_f1 = "n/a" if test_metrics is None else f"{test_metrics['f1']:.4f}"
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"test_loss={test_loss} test_oa={test_oa} test_f1={test_f1}"
        )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break
    training_seconds = time.time() - start_time
    return {
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }


@torch.no_grad()
def predict_full_dataset(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        x1, x2 = batch[0], batch[1]
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        output = _forward(model, x1, x2)
        predictions.extend(output.argmax(dim=1).cpu().numpy())
    return np.asarray(predictions)
