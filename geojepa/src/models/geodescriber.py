"""GeoDescriber: generates natural language descriptions from satellite embeddings."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class GeoDescriber(nn.Module):
    """
    Satellite image captioning model.
    Uses GPT-2 style decoder conditioned on vision embeddings + coordinates.
    """

    def __init__(
        self,
        embed_dim=384,
        vocab_size=50257,  # GPT-2 vocab
        max_length=256,
        num_layers=6,
        num_heads=6,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Vision embedding projection to decoder dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Coordinate encoding (lat, lon -> embedding)
        self.coord_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # GPT-2 style decoder
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embed_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length + 10,  # +10 for special tokens
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout
        )
        self.decoder = GPT2LMHeadModel(config)

        # Special tokens
        self.vision_token_id = vocab_size  # Vision embedding token
        self.coord_token_id = vocab_size + 1  # Coordinate token

        # Extend embedding table for special tokens
        self._resize_embeddings(vocab_size + 2)

    def _resize_embeddings(self, new_vocab_size):
        """Resize token embeddings to accommodate special tokens."""
        old_embeddings = self.decoder.transformer.wte
        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape

        new_embeddings = nn.Embedding(new_vocab_size, old_embedding_dim)
        new_embeddings.weight.data[:old_num_tokens] = old_embeddings.weight.data
        nn.init.normal_(new_embeddings.weight.data[old_num_tokens:], std=0.02)

        self.decoder.transformer.wte = new_embeddings
        self.decoder.lm_head = nn.Linear(old_embedding_dim, new_vocab_size, bias=False)
        self.decoder.lm_head.weight = new_embeddings.weight

    def forward(self, vision_embedding, coordinates, input_ids=None, labels=None):
        """
        Args:
            vision_embedding: (B, embed_dim) from vision encoder
            coordinates: (B, 2) [lat, lon] in degrees
            input_ids: (B, seq_len) token IDs for teacher forcing
            labels: (B, seq_len) target token IDs for loss computation

        Returns:
            loss if labels provided, else logits
        """
        B = vision_embedding.shape[0]

        # Project vision embedding
        vision_proj = self.vision_proj(vision_embedding)  # (B, embed_dim)

        # Encode coordinates
        coord_embed = self.coord_embed(coordinates)  # (B, embed_dim)

        # Combine: [VISION] vision_embed [COORD] coord_embed [BOS] text...
        # For simplicity, we prepend vision and coordinate embeddings as "soft prompts"

        if input_ids is not None:
            # Get token embeddings
            token_embeds = self.decoder.transformer.wte(input_ids)  # (B, seq_len, embed_dim)

            # Prepend vision and coordinate embeddings
            prefix_embeds = torch.stack([vision_proj, coord_embed], dim=1)  # (B, 2, embed_dim)
            full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (B, 2+seq_len, embed_dim)

            # Create attention mask: prefix tokens can attend to each other and all text tokens
            prefix_mask = torch.ones(B, 2, device=input_ids.device)
            if hasattr(self.decoder.transformer, 'attention_mask'):
                # Use existing attention mask logic
                pass

            # Forward through decoder
            outputs = self.decoder(inputs_embeds=full_embeds, labels=labels)
            return outputs.loss if labels is not None else outputs.logits
        else:
            # Generation mode
            return self.generate(vision_proj, coord_embed)

    def generate(self, vision_proj, coord_embed, max_length=256, temperature=1.0, top_p=0.9):
        """
        Autoregressive generation of description.

        Args:
            vision_proj: (B, embed_dim)
            coord_embed: (B, embed_dim)
            max_length: maximum generation length
            temperature: sampling temperature
            top_p: nucleus sampling parameter

        Returns:
            generated_ids: (B, generated_length)
        """
        B = vision_proj.shape[0]
        device = vision_proj.device

        # Start with vision and coordinate embeddings
        prefix_embeds = torch.stack([vision_proj, coord_embed], dim=1)  # (B, 2, embed_dim)

        # Start token (BOS = 50256 for GPT-2)
        bos_token = torch.full((B, 1), 50256, dtype=torch.long, device=device)
        generated = [bos_token]

        for _ in range(max_length):
            # Get embeddings for generated tokens
            if len(generated) > 1:
                gen_tokens = torch.cat(generated, dim=1)
                gen_embeds = self.decoder.transformer.wte(gen_tokens)
                full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
            else:
                full_embeds = torch.cat([prefix_embeds, self.decoder.transformer.wte(generated[0])], dim=1)

            # Forward pass
            outputs = self.decoder(inputs_embeds=full_embeds)
            logits = outputs.logits[:, -1, :] / temperature

            # Nucleus sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')

            # Sample next token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated.append(next_token)

            # Check for EOS (50256 for GPT-2)
            if (next_token == 50256).all():
                break

        return torch.cat(generated, dim=1)


class GeoDescriptionDataset:
    """
    Dataset for training GeoDescriber.
    Generates synthetic descriptions from OSM data.
    """

    TEMPLATES = [
        "{biome} landscape with {water_features}. {elevation_desc}. {infrastructure}.",
        "Region characterized by {biome}. {water_desc}. {urban_desc}.",
        "Satellite view of {biome} terrain. {elevation_desc}. {road_desc}.",
        "{climate_zone} area with {vegetation}. {water_features}. {settlement_desc}.",
    ]

    @staticmethod
    def generate_description(osm_tags, elevation, coordinates):
        """
        Generate synthetic description from OSM tags and elevation.

        Args:
            osm_tags: dict with OSM feature tags
            elevation: float, meters above sea level
            coordinates: (lat, lon)

        Returns:
            description string
        """
        # Determine biome from coordinates and tags
        lat = coordinates[0]

        if lat > 60 or lat < -60:
            biome = "polar/tundra"
        elif lat > 45 or lat < -45:
            biome = "boreal forest/taiga" if elevation < 1000 else "alpine/mountain"
        elif lat > 23 or lat < -23:
            biome = "temperate forest" if "forest" in str(osm_tags) else "grassland/steppe"
        else:
            biome = "tropical" if "forest" in str(osm_tags) else "savanna/arid"

        # Water features
        water_features = []
        if "water" in osm_tags:
            water_features.append("significant water bodies")
        if "river" in osm_tags:
            water_features.append("river systems")
        if "wetland" in osm_tags:
            water_features.append("wetland areas")
        water_str = ", ".join(water_features) if water_features else "no major water bodies"

        # Elevation
        if elevation > 2000:
            elevation_desc = f"High elevation terrain ({elevation:.0f}m)"
        elif elevation > 500:
            elevation_desc = f"Moderate elevation ({elevation:.0f}m)"
        else:
            elevation_desc = f"Low-lying terrain ({elevation:.0f}m)"

        # Infrastructure
        infra = []
        if "highway" in osm_tags:
            infra.append("road network")
        if "building" in osm_tags:
            infra.append("urban development")
        if "railway" in osm_tags:
            infra.append("rail infrastructure")
        infra_str = ", ".join(infra) if infra else "minimal infrastructure"

        # Build description
        import random
        template = random.choice(GeoDescriptionDataset.TEMPLATES)

        description = template.format(
            biome=biome,
            water_features=water_str,
            elevation_desc=elevation_desc,
            infrastructure=infra_str,
            water_desc=water_str,
            urban_desc=infra_str,
            road_desc=infra_str,
            climate_zone=biome.split("/")[0],
            vegetation=biome,
            settlement_desc=infra_str
        )

        return description
