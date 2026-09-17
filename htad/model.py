from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class DomainAutoencoder(nn.Module):
    """Lightweight domain-specific autoencoder for flattened L x c patches."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder(patches)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(patches))


class PaperDomainAutoencoder(nn.Module):
    """Three-layer symmetric domain autoencoder operating on complete patches."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        middle = max(latent_dim, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, middle), nn.LeakyReLU(0.2), nn.LayerNorm(middle),
            nn.Linear(middle, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, middle), nn.LeakyReLU(0.2), nn.LayerNorm(middle),
            nn.Linear(middle, hidden_dim), nn.LeakyReLU(0.2), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder(patches)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(patches))


class FrozenSemanticEncoder(nn.Module):
    """Frozen token embeddings with mean pooling and optional projection."""

    def __init__(
        self,
        text,
        d_model,
        token_count,
        backend,
        model_path,
        project,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.provenance = {"backend": backend, "text": text}
        if backend == "gpt2":
            if not model_path:
                raise ValueError("--gpt2-model-path is required for the gpt2 semantic backend")
            try:
                from transformers import GPT2Model, GPT2Tokenizer
            except ImportError:
                from .gpt2_embeddings import gpt2_pooled_embeddings

                pooled, ids_by_text = gpt2_pooled_embeddings(Path(model_path), [text])
                ids = torch.tensor(ids_by_text[0], dtype=torch.long)
                token_embeddings = pooled[0].unsqueeze(0)
                tokenizer = None
                backbone = None
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                backbone = GPT2Model.from_pretrained(model_path)
                for parameter in backbone.parameters():
                    parameter.requires_grad = False

                ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
                with torch.no_grad():
                    token_embeddings = backbone.wte(ids).detach().float()
            self.provenance.update(
                {
                    "model_path": str(model_path),
                    "tokenizer_class": (
                        tokenizer.__class__.__name__ if tokenizer is not None else "GPT2BPETokenizer"
                    ),
                    "model_class": (
                        backbone.__class__.__name__ if backbone is not None else "FrozenWTEOnly"
                    ),
                    "token_ids": [int(value) for value in ids.cpu().tolist()],
                    "token_count": int(len(ids)),
                    "source_embedding_dim": int(token_embeddings.shape[1]),
                    "pooling": "mean",
                    "projection": "learnable-local" if project else "external-shared",
                }
            )
        elif backend == "hash":
            pieces = text.split() or ["domain"]
            count = max(1, int(token_count))
            rows = []
            for index in range(count):
                piece = pieces[index % len(pieces)]
                digest = hashlib.sha256((piece + "#" + str(index)).encode("utf-8")).digest()
                seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
                rng = np.random.RandomState(seed)
                rows.append(rng.normal(0.0, 1.0 / math.sqrt(d_model), d_model))
            token_embeddings = torch.from_numpy(np.asarray(rows, dtype=np.float32))
            self.provenance.update(
                {
                    "token_count": count,
                    "source_embedding_dim": d_model,
                    "pooling": "mean",
                    "projection": "learnable-local" if project else "external-shared",
                }
            )
        else:
            raise KeyError("semantic backend must be 'hash' or 'gpt2'")

        self.provenance["token_embedding_sha256"] = hashlib.sha256(
            token_embeddings.contiguous().cpu().numpy().tobytes()
        ).hexdigest()
        pooled = token_embeddings.mean(dim=0, keepdim=True)
        self.register_buffer("pooled_embedding", pooled)
        self.output_dim = int(d_model)
        self.projection = nn.Linear(int(pooled.shape[1]), d_model) if project else None

    def forward(self, batch_size: int) -> torch.Tensor:
        value = self.pooled_embedding
        if self.projection is not None:
            value = self.projection(value)
        return value.expand(batch_size, -1)


def causal_mask(query_length: int, key_length: int, device, dtype) -> torch.Tensor:
    """C_ij = 0 for j <= i and -inf otherwise."""
    row = torch.arange(query_length, device=device).unsqueeze(1)
    column = torch.arange(key_length, device=device).unsqueeze(0)
    mask = torch.zeros((query_length, key_length), device=device, dtype=dtype)
    return mask.masked_fill(column > row, float("-inf"))


class AttentionBlock(nn.Module):
    """Pre-normalized causal Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, causal: bool = True) -> None:
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x).transpose(0, 1)
        mask = None
        if self.causal:
            mask = causal_mask(len(normalized), len(normalized), x.device, x.dtype)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + self.dropout1(attended.transpose(0, 1))
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x


class CausalGlobalDecoderLayer(nn.Module):
    """Causal self-attention followed by causal encoder cross-attention."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        causal,
        use_cross_attention,
    ) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.use_cross_attention = bool(use_cross_attention)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_dropout = nn.Dropout(dropout)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor) -> torch.Tensor:
        query = self.self_norm(x).transpose(0, 1)
        self_mask = None
        if self.causal:
            self_mask = causal_mask(len(query), len(query), x.device, x.dtype)
        attended, _ = self.self_attention(
            query, query, query, attn_mask=self_mask, need_weights=False
        )
        x = x + self.self_dropout(attended.transpose(0, 1))

        if self.use_cross_attention:
            query = self.query_norm(x).transpose(0, 1)
            memory = self.memory_norm(encoder_memory).transpose(0, 1)
            cross_mask = None
            if self.causal:
                cross_mask = causal_mask(len(query), len(memory), x.device, x.dtype)
            attended, _ = self.cross_attention(
                query, memory, memory, attn_mask=cross_mask, need_weights=False
            )
            x = x + self.cross_dropout(attended.transpose(0, 1))
        x = x + self.ff_dropout(self.feed_forward(self.ff_norm(x)))
        return x


class CausalGlobalBackbone(nn.Module):
    """Shared feature fusion and causal-global latent reconstruction network."""

    def __init__(
        self,
        patch_count,
        latent_dim,
        n_heads,
        encoder_layers,
        decoder_layers,
        dropout,
        causal_constraint,
        use_global_cross_attention,
    ) -> None:
        super().__init__()
        self.patch_count = int(patch_count)
        self.latent_dim = int(latent_dim)
        self.causal_constraint = bool(causal_constraint)
        self.use_global_cross_attention = bool(use_global_cross_attention)
        self.fusion_projection = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.fusion_norm = nn.LayerNorm(self.latent_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_count, self.latent_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.causal_encoder = nn.ModuleList(
            [
                AttentionBlock(
                    self.latent_dim, n_heads, dropout, self.causal_constraint
                )
                for _ in range(encoder_layers)
            ]
        )
        self.global_decoder = nn.ModuleList(
            [
                CausalGlobalDecoderLayer(
                    self.latent_dim,
                    n_heads,
                    dropout,
                    causal=self.causal_constraint,
                    use_cross_attention=self.use_global_cross_attention,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, latent: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        fused = self.fusion_norm(
            self.fusion_projection(torch.cat([latent, semantic], dim=-1))
        )
        encoded = self.input_dropout(fused + self.position_embedding)
        for layer in self.causal_encoder:
            encoded = layer(encoded)
        decoded = encoded
        for layer in self.global_decoder:
            decoded = layer(decoded, encoded)
        return self.output_projection(decoded)


class HTAD(nn.Module):
    """Method-aligned patch-domain HTAD generator."""

    def __init__(
        self,
        input_dim,
        window_size,
        latent_dim,
        ae_hidden_dim,
        patch_len,
        patch_stride,
        d_model,
        n_heads,
        encoder_layers,
        decoder_layers,
        dropout,
        semantic_text,
        semantic_tokens,
        semantic_backend,
        gpt2_model_path,
        paper_faithful,
        allow_gpt2_projection,
        causal_constraint,
        use_global_cross_attention,
    ) -> None:
        super().__init__()
        del allow_gpt2_projection
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if patch_len > window_size:
            raise ValueError("patch_len cannot exceed window_size")
        if patch_stride > patch_len:
            raise ValueError("patch_stride cannot exceed patch_len because it leaves uncovered timestamps")
        if latent_dim != d_model:
            raise ValueError("the method uses one common dimension d; latent_dim must equal d_model")
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (self.window_size - self.patch_len) // self.patch_stride
        covered = (self.patch_count - 1) * self.patch_stride + self.patch_len
        if covered != self.window_size:
            raise ValueError("window_size, patch_len, and patch_stride must cover the complete window")
        self.semantic_token_count = int(semantic_tokens)
        self.paper_faithful = bool(paper_faithful)
        self.method_faithful = True

        autoencoder = PaperDomainAutoencoder if self.paper_faithful else DomainAutoencoder
        self.domain_autoencoder = autoencoder(
            self.patch_len * self.input_dim, self.latent_dim, ae_hidden_dim
        )
        self.semantic_encoder = None
        if semantic_tokens > 0:
            self.semantic_encoder = FrozenSemanticEncoder(
                semantic_text, self.latent_dim, semantic_tokens, semantic_backend, gpt2_model_path
            )
        self.backbone = CausalGlobalBackbone(
            self.patch_count,
            self.latent_dim,
            n_heads,
            encoder_layers,
            decoder_layers,
            dropout,
            causal_constraint=causal_constraint,
            use_global_cross_attention=use_global_cross_attention,
        )

    @property
    def causal_encoder(self):
        return self.backbone.causal_encoder

    @property
    def global_decoder(self):
        return self.backbone.global_decoder

    def freeze_domain_modules(self) -> None:
        self.domain_autoencoder.eval()
        for parameter in self.domain_autoencoder.parameters():
            parameter.requires_grad = False

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        # unfold gives [B, k, C, L]; p_j is [L, c].
        return x.unfold(1, self.patch_len, self.patch_stride).permute(0, 1, 3, 2).contiguous()

    def merge_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        merged = patches.new_zeros((batch, self.window_size, self.input_dim))
        counts = patches.new_zeros((1, self.window_size, 1))
        for index in range(self.patch_count):
            start = index * self.patch_stride
            merged[:, start : start + self.patch_len] += patches[:, index]
            counts[:, start : start + self.patch_len] += 1.0
        return merged / counts.clamp_min(1.0)

    def autoencode_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = self.patchify(x)
        flat = patches.reshape(patches.shape[0], self.patch_count, -1)
        latent = self.domain_autoencoder.encode(flat)
        reconstructed = self.domain_autoencoder.decode(latent).reshape_as(patches)
        return reconstructed, patches

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1] != self.window_size or x.shape[2] != self.input_dim:
            raise ValueError(
                "expected [batch, {0}, {1}], got {2}".format(
                    self.window_size, self.input_dim, tuple(x.shape)
                )
            )
        patches = self.patchify(x)
        flat_patches = patches.reshape(x.shape[0], self.patch_count, -1)
        latent = self.domain_autoencoder.encode(flat_patches)

        if self.semantic_encoder is None:
            semantic = torch.zeros_like(latent)
        else:
            domain_vector = self.semantic_encoder(x.shape[0])
            semantic = domain_vector.unsqueeze(1).expand(-1, self.patch_count, -1)
        reconstructed_latent = self.backbone(latent, semantic)
        reconstructed_patches = self.domain_autoencoder.decode(reconstructed_latent).reshape_as(patches)
        reconstructed = self.merge_patches(reconstructed_patches)
        return reconstructed, latent, reconstructed_latent


class MultiDomainHTAD(nn.Module):
    """Cross-domain model with domain-specific patch I/O and a shared G."""

    def __init__(
        self,
        domain_specs,
        window_size,
        latent_dim,
        ae_hidden_dim,
        patch_len,
        patch_stride,
        n_heads,
        encoder_layers,
        decoder_layers,
        dropout,
        semantic_tokens,
        semantic_backend,
        gpt2_model_path,
        semantic_source,
        paper_faithful,
        causal_constraint
        use_global_cross_attention,
    ) -> None:
        super().__init__()
        if not domain_specs:
            raise ValueError("domain_specs cannot be empty")
        if latent_dim % n_heads:
            raise ValueError("latent_dim must be divisible by n_heads")
        if patch_len > window_size or patch_stride > patch_len:
            raise ValueError("patch_len/patch_stride do not define valid temporal patches")
        self.window_size = int(window_size)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (self.window_size - self.patch_len) // self.patch_stride
        covered = (self.patch_count - 1) * self.patch_stride + self.patch_len
        if covered != self.window_size:
            raise ValueError("window_size, patch_len, and patch_stride must cover the complete window")
        self.latent_dim = int(latent_dim)
        self.semantic_token_count = int(semantic_tokens)
        if semantic_source not in {"text", "domain-id", "none"}:
            raise ValueError("semantic_source must be 'text', 'domain-id', or 'none'")
        if semantic_tokens <= 0 and semantic_source == "text":
            semantic_source = "none"
        self.semantic_source = semantic_source
        self.paper_faithful = bool(paper_faithful)
        self.method_faithful = True
        self.input_dims = {str(key): int(value[0]) for key, value in domain_specs.items()}
        self.domain_to_id = {
            str(key): index for index, key in enumerate(domain_specs.keys())
        }

        autoencoder = PaperDomainAutoencoder if self.paper_faithful else DomainAutoencoder
        self.domain_autoencoders = nn.ModuleDict(
            {
                str(key): autoencoder(
                    self.patch_len * int(input_dim), self.latent_dim, ae_hidden_dim
                )
                for key, (input_dim, _) in domain_specs.items()
            }
        )
        self.semantic_encoders = nn.ModuleDict()
        self.semantic_projection = None
        self.domain_embedding = None
        if self.semantic_source == "text":
            for key, (_, text) in domain_specs.items():
                self.semantic_encoders[str(key)] = FrozenSemanticEncoder(
                    text,
                    self.latent_dim,
                    semantic_tokens,
                    semantic_backend,
                    gpt2_model_path,
                    project=False,
                )
            source_dims = {
                int(encoder.pooled_embedding.shape[1]) for encoder in self.semantic_encoders.values()
            }
            if len(source_dims) != 1:
                raise ValueError("all domain descriptions must use one language embedding dimension")
            self.semantic_projection = nn.Linear(source_dims.pop(), self.latent_dim)
        elif self.semantic_source == "domain-id":
            # Direct d-dimensional trainable identity vector; no text encoder,
            # projection MLP, or change to the downstream fusion interface.
            self.domain_embedding = nn.Embedding(len(self.domain_to_id), self.latent_dim)
        self.backbone = CausalGlobalBackbone(
            self.patch_count,
            self.latent_dim,
            n_heads,
            encoder_layers,
            decoder_layers,
            dropout,
            causal_constraint=causal_constraint,
            use_global_cross_attention=use_global_cross_attention,
        )

    def _check_domain(self, domain: str) -> str:
        key = str(domain)
        if key not in self.domain_autoencoders:
            raise KeyError("unknown domain {!r}".format(domain))
        return key

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(1, self.patch_len, self.patch_stride).permute(0, 1, 3, 2).contiguous()

    def merge_patches(self, patches: torch.Tensor, input_dim: int) -> torch.Tensor:
        merged = patches.new_zeros((patches.shape[0], self.window_size, input_dim))
        counts = patches.new_zeros((1, self.window_size, 1))
        for index in range(self.patch_count):
            start = index * self.patch_stride
            merged[:, start : start + self.patch_len] += patches[:, index]
            counts[:, start : start + self.patch_len] += 1.0
        return merged / counts.clamp_min(1.0)

    def autoencode_patches(self, x: torch.Tensor, domain: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._check_domain(domain)
        patches = self.patchify(x)
        flat = patches.reshape(patches.shape[0], self.patch_count, -1)
        module = self.domain_autoencoders[key]
        reconstructed = module.decode(module.encode(flat)).reshape_as(patches)
        return reconstructed, patches

    def freeze_domain_modules(self) -> None:
        for module in self.domain_autoencoders.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def semantic_provenance(self, domain: str):
        key = self._check_domain(domain)
        if self.semantic_source == "domain-id":
            return {
                "backend": "learnable-domain-id",
                "domain_id": self.domain_to_id[key],
                "embedding_dim": self.latent_dim,
                "trainable": True,
                "projection": "none",
            }
        if key not in self.semantic_encoders:
            return None
        return self.semantic_encoders[key].provenance

    def forward(self, x: torch.Tensor, domain: str):
        key = self._check_domain(domain)
        input_dim = self.input_dims[key]
        if x.ndim != 3 or x.shape[1] != self.window_size or x.shape[2] != input_dim:
            raise ValueError(
                "expected [batch, {0}, {1}] for {2}, got {3}".format(
                    self.window_size, input_dim, key, tuple(x.shape)
                )
            )
        patches = self.patchify(x)
        flat_patches = patches.reshape(x.shape[0], self.patch_count, -1)
        domain_module = self.domain_autoencoders[key]
        latent = domain_module.encode(flat_patches)
        if self.semantic_source == "domain-id":
            domain_ids = torch.full(
                (x.shape[0],), self.domain_to_id[key], device=x.device, dtype=torch.long
            )
            domain_vector = self.domain_embedding(domain_ids)
            semantic = domain_vector.unsqueeze(1).expand(-1, self.patch_count, -1)
        elif key not in self.semantic_encoders:
            semantic = torch.zeros_like(latent)
        else:
            pooled = self.semantic_encoders[key](x.shape[0])
            domain_vector = self.semantic_projection(pooled)
            semantic = domain_vector.unsqueeze(1).expand(-1, self.patch_count, -1)
        reconstructed_latent = self.backbone(latent, semantic)
        reconstructed_patches = domain_module.decode(reconstructed_latent).reshape_as(patches)
        reconstructed = self.merge_patches(reconstructed_patches, input_dim)
        return reconstructed, latent, reconstructed_latent


class MLPDiscriminator(nn.Module):
    """Shared discriminator on the common d-dimensional patch interface."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        paper_faithful,
    ) -> None:
        super().__init__()
        half = max(8, hidden_dim // 2)
        layers = [
            nn.Linear(input_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(dropout),
        ]
        if paper_faithful:
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                 nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            )
        layers.extend(
            [nn.Linear(hidden_dim, half), nn.LayerNorm(half), nn.LeakyReLU(0.2),
             nn.Dropout(dropout) if paper_faithful else nn.Identity(), nn.Linear(half, 1)]
        )
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).squeeze(-1)

    def extract_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the penultimate discriminator representation for scoring."""
        return self.network[:-1](latent)


class HybridAuxiliaryHeads(nn.Module):
    """Learned latent mask token and shared next-patch forecasting head."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.forecast = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forecast_next(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forecast(latent)


class DatasetAwareCausalForecaster(nn.Module):
    """Strict-history forecaster with entity-specific I/O and a shared backbone."""

    def __init__(
        self,
        entity_specs,
        history_length,
        hidden_dim,
        semantic_dim,
        dropout,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.entity_specs = {
            str(key): (int(value[0]), int(value[1])) for key, value in entity_specs.items()
        }
        self.input_heads = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(self.history_length * input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for key, (input_dim, _) in self.entity_specs.items()
        })
        self.semantic_projection = nn.Linear(semantic_dim, hidden_dim)
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.output_heads = nn.ModuleDict({
            key: nn.Linear(hidden_dim, target_dim)
            for key, (_, target_dim) in self.entity_specs.items()
        })

    def forward(
        self, history: torch.Tensor, entity: str, semantic_vector: torch.Tensor
    ) -> torch.Tensor:
        key = str(entity)
        if key not in self.entity_specs:
            raise KeyError("unknown forecasting entity {!r}".format(entity))
        input_dim, _ = self.entity_specs[key]
        if history.ndim != 3 or history.shape[1:] != (self.history_length, input_dim):
            raise ValueError(
                "expected [batch, {}, {}] for {}, got {}".format(
                    self.history_length, input_dim, key, tuple(history.shape)
                )
            )
        if semantic_vector.ndim == 1:
            semantic_vector = semantic_vector.unsqueeze(0)
        conditioned = self.input_heads[key](history.reshape(history.shape[0], -1))
        conditioned = conditioned + self.semantic_projection(semantic_vector).expand_as(conditioned)
        return self.output_heads[key](self.shared_backbone(conditioned))
