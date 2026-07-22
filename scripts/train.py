#!/usr/bin/env python3
"""Main training script for GeoJEPA with auto-download."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import DEFAULT_CONFIG
from src.data.dataset import create_dataloader
from src.training.trainer import GeoJEPATrainer, DownstreamTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GeoJEPA model")
    parser.add_argument("--data-dir", type=str, default="data/satellite", help="Path to satellite imagery data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no download)")
    parser.add_argument("--auto-download", action="store_true", default=True, help="Auto-download if data missing")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to download/generate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, default=None, help="Load pretrained encoder for downstream")
    parser.add_argument("--downstream", action="store_true", help="Train downstream tasks only")
    parser.add_argument("--sh-client-id", type=str, default=None, help="Sentinel Hub client ID")
    parser.add_argument("--sh-client-secret", type=str, default=None, help="Sentinel Hub client secret")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()

    # Update config
    config = DEFAULT_CONFIG
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.num_workers = args.workers

    print("=" * 60)
    print("GEojEPA Training")
    print("=" * 60)
    print(f"Device: {config.training.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data: {'synthetic' if args.synthetic else args.data_dir}")
    print(f"Auto-download: {args.auto_download}")
    print("=" * 60)

    # Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_synthetic=args.synthetic,
        num_synthetic_samples=args.num_samples,
        auto_download=args.auto_download and not args.synthetic,
        sh_client_id=args.sh_client_id,
        sh_client_secret=args.sh_client_secret
    )

    val_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_synthetic=args.synthetic,
        num_synthetic_samples=max(args.num_samples // 5, 1000),
        auto_download=args.auto_download and not args.synthetic,
        sh_client_id=args.sh_client_id,
        sh_client_secret=args.sh_client_secret
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    if args.downstream and args.pretrained:
        # Train downstream tasks
        print("\n[2/4] Loading pretrained model...")
        from src.models.geojepa import GeoJEPA
        import torch

        model = GeoJEPA(config)
        checkpoint = torch.load(args.pretrained, map_location=config.training.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("[3/4] Training downstream tasks...")
        downstream_trainer = DownstreamTrainer(model, config, checkpoint_dir=args.checkpoint_dir)

        print("  -> Training GeoLocator...")
        downstream_trainer.train_locator(train_loader, epochs=10)

        print("  -> Training GeoDescriber...")
        downstream_trainer.train_describer(train_loader, epochs=10)

        downstream_trainer.save_heads()
        print("[4/4] Downstream training complete!")

    else:
        # Pretrain JEPA
        print("\n[2/4] Initializing GeoJEPA trainer...")
        trainer = GeoJEPATrainer(config, checkpoint_dir=args.checkpoint_dir)

        if args.resume:
            print(f"[3/4] Resuming from {args.resume}...")
            trainer.load_checkpoint(args.resume)
        else:
            print("[3/4] Starting pretraining from scratch...")

        print(f"[4/4] Training for {args.epochs} epochs...")
        print("-" * 60)
        trainer.train(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {Path(args.checkpoint_dir).absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
