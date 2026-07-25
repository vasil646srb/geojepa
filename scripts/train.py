#!/usr/bin/env python3
"""
GeoJEPA Training — с автозагрузкой реальных HLS-данных через NASA Earthdata.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.hls_downloader import HLSDownloader
from src.data.real_dataset import GeoImageDataset
from src.training.trainer import GeoJEPATrainer
from src.models.geojepa import GeoJEPA


def get_dataloaders(args):
    """Создаёт даталоадеры: при необходимости сначала скачивает данные."""
    # 1. Автозагрузка реальных снимков
    if args.auto_download:
        print("\n" + "="*60)
        print("📡 Автозагрузка реальных данных HLS")
        print("="*60)
        downloader = HLSDownloader(output_dir=args.data_dir)
        downloader.download(
            n_samples=args.num_samples,
            temporal=(args.temporal_start, args.temporal_end)
        )

    # 2. Создаём датасет из .npz файлов
    print("\n[1/4] Создание датасета...")
    full_dataset = GeoImageDataset(data_dir=args.data_dir)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"❌ Нет данных в {args.data_dir}. "
            f"Запустите с --auto-download или укажите правильный --data-dir"
        )

    print(f"   Всего сэмплов: {len(full_dataset)}")

    # 3. Train / Val split (80/20)
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
    # Data
    parser.add_argument('--data-dir', type=str, default='data/hls_real', help='Папка с .npz снимками')
    parser.add_argument('--auto-download', action='store_true', help='Автоматически скачать недостающие снимки')
    parser.add_argument('--num-samples', type=int, default=1000, help='Целевое количество сэмплов')
    parser.add_argument('--temporal-start', type=str, default='2025-05-01', help='Начало периода HLS')
    parser.add_argument('--temporal-end', type=str, default='2025-08-31', help='Конец периода HLS')
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Путь к чекпоинту для продолжения')
    args = parser.parse_args()

    print("="*60)
    print("🌍 GeoJEPA Training")
    print("="*60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data dir: {args.data_dir}")
    print("="*60)

    # Создаём даталоадеры
    train_loader, val_loader = get_dataloaders(args)

    # Инициализация модели и тренера
    print("\n[2/4] Инициализация модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeoJEPA().to(device)

    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Тренер
    print("[3/4] Инициализация тренера...")
    trainer = GeoJEPATrainer(model, optimizer, scheduler, device, args.checkpoint_dir)

    # Возобновление
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"[4/4] Возобновление с {args.resume}...")
        start_epoch = trainer.load_checkpoint(args.resume)
    else:
        print("[4/4] Обучение с нуля...")

    # Цикл обучения
    print("\n" + "-"*60)
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val MAE (км): {val_metrics.get('mae_km', 'N/A')}")

        # Сохраняем лучший чекпоинт
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

