#!/usr/bin/env python3
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
from torch.utils.data import DataLoader, random_split

from src.data.hls_downloader import HLSDownloader
from src.data.real_dataset import GeoImageDataset

try:
    from src.training.trainer import GeoJEPATrainer as Trainer
except ImportError:
    from src.training.trainer import Trainer

try:
    from src.models.geojepa import GeoJEPA
except ImportError:
    from src.models.geojepa import GeoJEPA


def get_dataloaders(args):
    if args.auto_download:
        print("\n" + "="*60)
        print("📡 Автозагрузка реальных данных HLS")
        print("="*60)
        downloader = HLSDownloader(output_dir=args.data_dir)
        downloader.download(
            n_samples=args.num_samples,
            temporal=(args.temporal_start, args.temporal_end)
        )

    print("\n[1/4] Создание датасета...")
    full_dataset = GeoImageDataset(data_dir=args.data_dir)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"❌ Нет данных в {args.data_dir}. "
            f"Запустите с --auto-download или укажите правильный --data-dir"
        )

    print(f"   Всего сэмплов: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {args.batch_size}")
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="GeoJEPA Training")
    parser.add_argument('--data-dir', type=str, default='data/hls_real')
    parser.add_argument('--auto-download', action='store_true')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--temporal-start', type=str, default='2025-05-01')
    parser.add_argument('--temporal-end', type=str, default='2025-08-31')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    print("="*60)
    print("🌍 GeoJEPA Training")
    print("="*60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data dir: {args.data_dir}")
    print("="*60)

    train_loader, val_loader = get_dataloaders(args)

    print("\n[2/4] Инициализация модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeoJEPA().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("[3/4] Инициализация тренера...")
    trainer = Trainer(model, optimizer, scheduler, device, args.checkpoint_dir)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"[4/4] Возобновление с {args.resume}...")
        start_epoch = trainer.load_checkpoint(args.resume)
    else:
        print("[4/4] Обучение с нуля...")

    print("\n" + "-"*60)
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_checkpoint(epoch, is_best=True)
            print(f"   💾 Новый лучший чекпоинт!")

    print("\n" + "="*60)
    print("✅ Обучение завершено!")
    print(f"Лучший чекпоинт: {args.checkpoint_dir}/best_model.pt")
    print("="*60)


if __name__ == "__main__":
    main()
