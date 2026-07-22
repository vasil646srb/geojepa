#!/usr/bin/env python3
"""
GeoJEPA Quickstart Script
One-command setup and training for RTX 3090 24GB.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd, desc=None):
    """Run shell command with logging."""
    if desc:
        print(f"\n{'='*60}")
        print(f"{desc}")
        print(f"{'='*60}")
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed with code {result.returncode}")
    return result.returncode == 0


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    GeoJEPA Quickstart                         ║
    ║         RTX 3090 24GB Optimized Training Pipeline             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check GPU
    print("Checking GPU...")
    run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "GPU Info")

    # Install dependencies
    print("\n[1/6] Installing dependencies...")
    run_cmd("pip install -e .", "Installing GeoJEPA")
    run_cmd("pip install -e '.[all]'", "Installing optional dependencies")

    # Quick test with synthetic data (5 min)
    print("\n[2/6] Quick test with synthetic data...")
    run_cmd(
        "python scripts/train.py --synthetic --num-samples 1000 --epochs 2 --batch-size 64",
        "Quick test (2 epochs, synthetic)"
    )

    # Download real data (optional, requires Sentinel Hub credentials)
    print("\n[3/6] Downloading satellite data...")
    print("NOTE: For real Sentinel-2 data, set SH_CLIENT_ID and SH_CLIENT_SECRET env vars")
    print("      Get credentials at: https://dataspace.copernicus.eu/")
    print("      Falling back to synthetic data if credentials not provided...")

    sh_id = os.environ.get("SH_CLIENT_ID", "")
    sh_secret = os.environ.get("SH_CLIENT_SECRET", "")

    if sh_id and sh_secret:
        run_cmd(
            f"python scripts/train.py --auto-download --num-samples 10000 "
            f"--sh-client-id {sh_id} --sh-client-secret {sh_secret} "
            f"--epochs 10 --batch-size 64",
            "Pretraining with real data (10 epochs)"
        )
    else:
        print("No Sentinel Hub credentials found. Using synthetic data...")
        run_cmd(
            "python scripts/train.py --synthetic --num-samples 10000 "
            "--epochs 10 --batch-size 64",
            "Pretraining with synthetic data (10 epochs)"
        )

    # Full pretraining (100 epochs)
    print("\n[4/6] Full JEPA pretraining...")
    print("Estimated time on RTX 3090: ~50 hours for 100 epochs with 100K samples")
    print("Run this manually when ready:")
    print("  python scripts/train.py --synthetic --num-samples 100000 --epochs 100 --batch-size 64")

    # Downstream training
    print("\n[5/6] Downstream task training...")
    print("After pretraining completes, run:")
    print("  python scripts/train.py --downstream --pretrained checkpoints/best_model.pt")

    # Demo
    print("\n[6/6] Launching demo...")
    print("After downstream training, run:")
    print("  python scripts/demo.py --checkpoint checkpoints/best_model.pt")

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      Quickstart Complete!                     ║
    ╚═══════════════════════════════════════════════════════════════╝

    Next steps:
    1. Get Sentinel Hub credentials for real data
    2. Run full pretraining: 100 epochs, ~50 hours
    3. Train downstream tasks: ~4 hours
    4. Launch demo and test!

    Total estimated time (RTX 3090 24GB):
    - Synthetic test: 5 minutes
    - Full pretraining: 50 hours
    - Downstream: 4 hours
    - TOTAL: ~54 hours
    """)


if __name__ == "__main__":
    main()
