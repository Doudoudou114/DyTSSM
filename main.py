import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from load_data import (
    apply_pca,
    generate_data,
    load_data,
    normalize_image,
    visualize_predictions,
)
from model import DyTSSM
from train_test import evaluate, predict_full_dataset, run_training


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate DyTSSM.")
    parser.add_argument("--dataset", default="hermiston")
    parser.add_argument("--data-root", default=str(project_root.parent / "dataset"))
    parser.add_argument("--output-dir", default=str(project_root / "output"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pca-channels", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--label-ratio", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--attn-drop-rate", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_test_predictions(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for x1, x2, target in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            output = model(torch.cat([x1, x2], dim=1))
            labels.extend(target.numpy())
            predictions.extend(output.argmax(dim=1).cpu().numpy())
    return labels, predictions


def save_results(output_path, config, training_result, metrics, labels, predictions):
    serializable_metrics = {
        key: float(value) for key, value in metrics.items() if key != "confusion"
    }
    summary = {
        "config": config,
        "training": {
            "training_seconds": training_result["training_seconds"],
            "best_epoch": training_result["best_epoch"],
            "best_score": training_result["best_score"],
        },
        "test": serializable_metrics,
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (output_path / "results.txt").open("w", encoding="utf-8") as file:
        file.write(f"dataset: {config['dataset']}\n")
        file.write(f"best_epoch: {training_result['best_epoch']}\n")
        file.write(f"training_seconds: {training_result['training_seconds']:.2f}\n")
        for key, value in serializable_metrics.items():
            file.write(f"{key}: {value:.4f}\n")
        file.write("\n")
        file.write(
            classification_report(
                labels,
                predictions,
                labels=[0, 1],
                target_names=["unchanged", "changed"],
                zero_division=0,
            )
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output_dir) / args.dataset
    output_path.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)
    image1, image2, labels = load_data(args.dataset, args.data_root)
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    image1 = apply_pca(image1, args.pca_channels)
    image2 = apply_pca(image2, args.pca_channels)
    data = generate_data(
        image1,
        image2,
        labels,
        args.batch_size,
        args.label_ratio,
        device,
        args.patch_size,
        args.noise_std,
    )

    model = DyTSSM(
        in_chans=args.pca_channels,
        num_classes=2,
        dim=args.d_model,
        depth=args.depth,
        d_state=args.d_state,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=data.class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )
    best_model_path = output_path / "best_model.pth"
    print(
        f"dataset={args.dataset} train={data.train_size} "
        f"test={data.test_size} device={device}"
    )
    training_result = run_training(
        model=model,
        train_loader=data.train_loader,
        test_loader=data.test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        best_model_path=best_model_path,
        validate_every=args.validate_every,
        patience=args.patience,
    )
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, data.test_loader, criterion, device)
    test_labels, test_predictions = collect_test_predictions(model, data.test_loader, device)
    predictions = predict_full_dataset(model, data.full_loader, device)
    visualization_path = output_path / "prediction.png"
    visualize_predictions(labels, predictions, data.full_indices, visualization_path)
    save_results(
        output_path,
        vars(args),
        training_result,
        metrics,
        test_labels,
        test_predictions,
    )
    print(
        f"test_loss={metrics['loss']:.4f} oa={metrics['oa']:.4f} "
        f"f1={metrics['f1']:.4f} kappa={metrics['kappa']:.4f}"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
