"""Training script for GeoJEPA pretraining and fine-tuning."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from pathlib import Path
import json

from ..models.geojepa import GeoJEPA
from ..models.geolocator import GeoLocator, haversine_loss, grid_to_coordinates
from ..models.geodescriber import GeoDescriber
from ..data.dataset import create_dataloader
from ..config import DEFAULT_CONFIG


class GeoJEPATrainer:
    """Trainer for GeoJEPA model."""

    def __init__(self, config=None, checkpoint_dir="checkpoints"):
        self.config = config or DEFAULT_CONFIG
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(self.config.training.device if torch.cuda.is_available() else "cpu")

        # Model
        self.model = GeoJEPA(self.config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            betas=self.config.training.betas
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if self.config.training.mixed_precision else None

        # Metrics
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.training.warmup_epochs:
                return step / self.config.training.warmup_epochs
            else:
                # Cosine decay
                progress = (step - self.config.training.warmup_epochs) /                           (self.config.training.epochs - self.config.training.warmup_epochs)
                return self.config.training.min_lr / self.config.training.lr +                        (1 - self.config.training.min_lr / self.config.training.lr) *                        0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_jepa = 0.0
        total_contrastive = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            coordinates = batch["coordinates"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(images, coordinates)
                    loss = outputs["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, coordinates)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

            # Update EMA target encoder
            self.model.update_target_encoder()

            # Metrics
            total_loss += loss.item()
            total_jepa += outputs["jepa_loss"].item()
            total_contrastive += outputs["contrastive_loss"].item() if isinstance(outputs["contrastive_loss"], torch.Tensor) else outputs["contrastive_loss"]

            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "jepa": f"{outputs['jepa_loss'].item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        self.scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_jepa = total_jepa / len(dataloader)
        avg_contrastive = total_contrastive / len(dataloader)

        return {
            "loss": avg_loss,
            "jepa_loss": avg_jepa,
            "contrastive_loss": avg_contrastive
        }

    def validate(self, dataloader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch["image"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)

                outputs = self.model(images, coordinates)
                total_loss += outputs["loss"].item()

        return {"loss": total_loss / len(dataloader)}

    def train(self, train_loader, val_loader=None, epochs=None):
        """Full training loop."""
        epochs = epochs or self.config.training.epochs

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                  f"jepa={train_metrics['jepa_loss']:.4f}, "
                  f"contrastive={train_metrics['contrastive_loss']:.4f}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                print(f"Validation loss: {val_metrics['loss']:.4f}")

            # Save checkpoint
            if epoch % self.config.training.save_every == 0 or epoch == epochs - 1:
                self.save_checkpoint()

            if train_metrics["loss"] < self.best_loss:
                self.best_loss = train_metrics["loss"]
                self.save_checkpoint("best_model.pt")

    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
        filename = filename or f"checkpoint_epoch_{self.epoch}.pt"
        path = self.checkpoint_dir / filename

        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config
        }, path)

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        print(f"Checkpoint loaded: {path}")


class DownstreamTrainer:
    """Trainer for downstream tasks (geolocation and description)."""

    def __init__(self, geojepa_model, config=None, checkpoint_dir="checkpoints/downstream"):
        self.config = config or DEFAULT_CONFIG
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(self.config.training.device if torch.cuda.is_available() else "cpu")

        # Freeze encoder
        self.encoder = geojepa_model.context_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Task heads
        self.geo_locator = GeoLocator(
            embed_dim=self.config.model.embed_dim,
            num_continent_classes=self.config.model.num_continent_classes,
            num_country_classes=self.config.model.num_country_classes,
            num_region_classes=self.config.model.num_region_classes,
            num_grid_classes=self.config.model.num_grid_classes,
            use_hierarchy=True
        ).to(self.device)

        self.geo_describer = GeoDescriber(
            embed_dim=self.config.model.embed_dim,
            max_length=self.config.model.max_description_length
        ).to(self.device)

        # Optimizers
        self.locator_optimizer = torch.optim.AdamW(
            self.geo_locator.parameters(),
            lr=self.config.training.lr * 0.1,
            weight_decay=self.config.training.weight_decay
        )

        self.describer_optimizer = torch.optim.AdamW(
            self.geo_describer.parameters(),
            lr=self.config.training.lr * 0.1,
            weight_decay=self.config.training.weight_decay
        )

        self.scaler = GradScaler() if self.config.training.mixed_precision else None

    def train_locator(self, dataloader, epochs=10):
        """Train geolocation head."""
        self.geo_locator.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_distance = 0.0

            pbar = tqdm(dataloader, desc=f"Locator Epoch {epoch}")
            for batch in pbar:
                images = batch["image"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)

                # Get embeddings
                with torch.no_grad():
                    embeddings = self.encoder(images)

                # Forward
                self.locator_optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        outputs = self.geo_locator(features=embeddings)
                        # Hierarchical loss
                        loss = 0
                        for level in ["continent", "country", "region", "grid"]:
                            loss += nn.functional.cross_entropy(outputs[level], self._coords_to_labels(coordinates, level))

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.locator_optimizer)
                    self.scaler.update()
                else:
                    outputs = self.geo_locator(features=embeddings)
                    loss = 0
                    for level in ["continent", "country", "region", "grid"]:
                        loss += nn.functional.cross_entropy(outputs[level], self._coords_to_labels(coordinates, level))
                    loss.backward()
                    self.locator_optimizer.step()

                total_loss += loss.item()

                # Compute approximate distance error
                pred_grid = torch.argmax(outputs["grid"], dim=-1)
                pred_coords = grid_to_coordinates(pred_grid)
                dist = haversine_loss(pred_coords, coordinates)
                total_distance += dist.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dist_km": f"{dist.item():.1f}"
                })

            print(f"Locator Epoch {epoch}: loss={total_loss/len(dataloader):.4f}, "
                  f"avg_dist={total_distance/len(dataloader):.1f}km")

    def _coords_to_labels(self, coordinates, level):
        """Convert coordinates to class labels for hierarchical training."""
        # Simplified: assign based on coordinate ranges
        lat, lon = coordinates[:, 0], coordinates[:, 1]

        if level == "continent":
            # 7 continents based on lat/lon
            labels = torch.zeros(len(coordinates), dtype=torch.long, device=coordinates.device)
            labels[(lat > 35) & (lon > -30) & (lon < 60)] = 0  # Europe/Asia
            labels[(lat > 10) & (lon > -20) & (lon < 55)] = 1  # Africa
            labels[(lat > 10) & (lon > 55) & (lon < 180)] = 2  # Asia
            labels[(lat > 15) & (lon > -140) & (lon < -50)] = 3  # North America
            labels[(lat < 10) & (lon > -90) & (lon < -30)] = 4  # South America
            labels[(lat < -10) & (lon > 110) & (lon < 180)] = 5  # Australia/Oceania
            labels[lat < -60] = 6  # Antarctica
            return labels

        elif level == "grid":
            # 1° x 1° grid cells
            lat_idx = ((lat + 90) / 1).long().clamp(0, 179)
            lon_idx = ((lon + 180) / 1).long().clamp(0, 359)
            return lat_idx * 360 + lon_idx

        else:
            # Simplified for country/region
            return torch.zeros(len(coordinates), dtype=torch.long, device=coordinates.device)

    def train_describer(self, dataloader, epochs=10):
        """Train description generation head."""
        self.geo_describer.train()

        for epoch in range(epochs):
            total_loss = 0.0

            pbar = tqdm(dataloader, desc=f"Describer Epoch {epoch}")
            for batch in pbar:
                images = batch["image"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)

                # Get embeddings
                with torch.no_grad():
                    embeddings = self.encoder(images)

                # Generate synthetic descriptions (in real case, load from OSM)
                from ..models.geodescriber import GeoDescriptionDataset
                descriptions = []
                for i in range(len(coordinates)):
                    desc = GeoDescriptionDataset.generate_description(
                        {}, 
                        batch["elevation"][i].item(),
                        (coordinates[i, 0].item(), coordinates[i, 1].item())
                    )
                    descriptions.append(desc)

                # Tokenize descriptions (simplified)
                # In practice, use proper tokenizer
                self.describer_optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        # Dummy loss for demonstration
                        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.describer_optimizer)
                    self.scaler.update()
                else:
                    loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    loss.backward()
                    self.describer_optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"Describer Epoch {epoch}: loss={total_loss/len(dataloader):.4f}")

    def save_heads(self):
        """Save trained heads."""
        torch.save({
            "locator": self.geo_locator.state_dict(),
            "describer": self.geo_describer.state_dict()
        }, self.checkpoint_dir / "downstream_heads.pt")
        print("Downstream heads saved")
