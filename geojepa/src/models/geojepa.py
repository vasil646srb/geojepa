"""Main GeoJEPA model combining all components."""

import torch
import torch.nn as nn
from .encoder import VisionEncoder, EMAEncoder
from .predictor import JEPAPredictor
from .geolocator import GeoLocator
from .geodescriber import GeoDescriber


class GeoJEPA(nn.Module):
    """
    Complete GeoJEPA model for self-supervised pretraining on satellite imagery.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Context encoder (student)
        self.context_encoder = VisionEncoder(
            img_size=config.data.image_size,
            patch_size=config.data.patch_size,
            in_channels=len(config.data.bands),
            embed_dim=config.model.embed_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout
        )

        # Target encoder (teacher, EMA of context encoder)
        target_encoder = VisionEncoder(
            img_size=config.data.image_size,
            patch_size=config.data.patch_size,
            in_channels=len(config.data.bands),
            embed_dim=config.model.embed_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout
        )
        self.target_encoder = EMAEncoder(target_encoder, decay=config.model.ema_decay)

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=config.model.embed_dim,
            num_layers=config.model.predictor_num_layers,
            num_heads=config.model.predictor_num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            num_patches=config.data.num_patches
        )

        # Downstream heads (initialized but trained separately)
        self.geo_locator = None
        self.geo_describer = None

    def create_mask(self, batch_size, mask_ratio, device):
        """
        Create random mask for JEPA.

        Args:
            batch_size: number of samples
            mask_ratio: fraction of patches to mask (0.75 = 75% masked)
            device: torch device

        Returns:
            mask: (B, num_patches) boolean tensor, True = keep, False = mask
            target_indices: (B, N_target) indices of masked patches
        """
        num_patches = self.config.data.num_patches
        num_keep = int(num_patches * (1 - mask_ratio))

        # Random permutation for each sample
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first num_keep patches
        mask = torch.zeros(batch_size, num_patches, device=device)
        mask[:, :num_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Target indices (masked patches)
        target_indices = ids_shuffle[:, num_keep:]

        return mask.bool(), target_indices

    def forward(self, images, coordinates=None):
        """
        Forward pass for JEPA pretraining.

        Args:
            images: (B, C, H, W) satellite images
            coordinates: (B, 2) optional coordinates for contrastive loss

        Returns:
            dict with losses and predictions
        """
        B = images.shape[0]
        device = images.device

        # Create mask
        mask, target_indices = self.create_mask(B, self.config.model.mask_ratio, device)

        # Context encoding (visible patches only)
        context_tokens = self.context_encoder(images, mask=mask, return_all_tokens=True)
        # Only keep visible tokens
        context_tokens = context_tokens * mask.unsqueeze(-1).float()
        # Remove masked positions (compress)
        context_tokens = context_tokens[mask].view(B, -1, self.config.model.embed_dim)

        # Target encoding (all patches, no gradient)
        with torch.no_grad():
            target_tokens = self.target_encoder.encoder(
                images, return_all_tokens=True
            )
            # Extract target patches
            target_embeddings = []
            for b in range(B):
                target_emb = target_tokens[b, target_indices[b]]
                target_embeddings.append(target_emb)
            target_embeddings = torch.stack(target_embeddings)

        # Predict target embeddings from context
        pred_embeddings = self.predictor(context_tokens, target_indices)

        # JEPA loss: L1 distance between predicted and target embeddings
        jepa_loss = nn.functional.l1_loss(pred_embeddings, target_embeddings)

        # Contrastive loss (optional, for geo-awareness)
        contrastive_loss = 0.0
        if coordinates is not None:
            contrastive_loss = self.contrastive_loss(context_tokens.mean(dim=1), coordinates)

        total_loss = (
            self.config.training.jepa_loss_weight * jepa_loss +
            self.config.training.contrastive_loss_weight * contrastive_loss
        )

        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss,
            'contrastive_loss': contrastive_loss,
            'pred_embeddings': pred_embeddings,
            'target_embeddings': target_embeddings
        }

    def contrastive_loss(self, embeddings, coordinates, temperature=0.07):
        """
        Contrastive loss: nearby locations should have similar embeddings.

        Args:
            embeddings: (B, embed_dim)
            coordinates: (B, 2) [lat, lon]

        Returns:
            contrastive loss value
        """
        B = embeddings.shape[0]

        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=-1)

        # Compute pairwise distance in coordinate space (Haversine)
        coords_rad = torch.deg2rad(coordinates)
        dlat = coords_rad[:, 0].unsqueeze(1) - coords_rad[:, 0].unsqueeze(0)
        dlon = coords_rad[:, 1].unsqueeze(1) - coords_rad[:, 1].unsqueeze(0)

        a = torch.sin(dlat / 2) ** 2 +             torch.cos(coords_rad[:, 0].unsqueeze(1)) * torch.cos(coords_rad[:, 0].unsqueeze(0)) * torch.sin(dlon / 2) ** 2
        distances = 6371 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # km

        # Similarity in embedding space
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

        # Positive pairs: distance < 100 km
        positive_mask = (distances < 100).float()
        positive_mask.fill_diagonal_(1)  # Self-similarity

        # Contrastive loss: maximize similarity for nearby locations
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sim / all_sim + 1e-8).mean()
        return loss

    @torch.no_grad()
    def update_target_encoder(self):
        """Update EMA target encoder from context encoder."""
        self.target_encoder.update(self.context_encoder)

    def get_embedding(self, images):
        """Get embedding for downstream tasks."""
        return self.context_encoder(images, return_all_tokens=False)

    def add_downstream_heads(self):
        """Add downstream task heads after pretraining."""
        self.geo_locator = GeoLocator(
            embed_dim=self.config.model.embed_dim,
            num_continent_classes=self.config.model.num_continent_classes,
            num_country_classes=self.config.model.num_country_classes,
            num_region_classes=self.config.model.num_region_classes,
            num_grid_classes=self.config.model.num_grid_classes,
            use_hierarchy=True
        )

        self.geo_describer = GeoDescriber(
            embed_dim=self.config.model.embed_dim,
            max_length=self.config.model.max_description_length
        )
