from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import evaluation_windows
from .metrics import anomaly_scores


def seed_everything(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_reconstruction_errors(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """R_i = mean over the T timestamps and c domain variables."""
    return torch.square(reconstructed - real).mean(dim=(1, 2))


def robust_error_statistics(errors: torch.Tensor, epsilon: float = 1e-6) -> Tuple[float, float]:
    values = errors.detach().float().reshape(-1)
    median = torch.median(values)
    mad = torch.median(torch.abs(values - median))
    scale = 1.4826 * mad + float(epsilon)
    return float(median.cpu()), float(scale.cpu())


def reliability_weights(
    sample_errors: torch.Tensor,
    previous_median: Optional[float] = None,
    previous_scale: Optional[float] = None,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Independent bounded sample weights from previous-epoch median/MAD."""
    with torch.no_grad():
        if previous_median is None or previous_scale is None or alpha <= 0.0:
            return torch.ones_like(sample_errors)
        scale = max(float(previous_scale), 1e-12)
        z_score = (sample_errors - float(previous_median)) / scale
        reliability = torch.exp(-float(beta) * torch.clamp(z_score, min=0.0))
        strength = min(1.0, max(0.0, float(alpha)))
        return (1.0 - strength) + strength * reliability


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Equation-level sum(w_i value_i) / (sum(w_i) + xi)."""
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be equal-length sample vectors")
    return torch.sum(values * weights) / (torch.sum(weights) + float(epsilon))


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


class HTADTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        adversarial_weight: float = 0.01,
        weight_decay: float = 0.0,
        gradient_clip: float = 5.0,
        use_adaptive_weight: bool = True,
        use_gan: bool = True,
        warmup_epochs: int = 3,
        ramp_epochs: int = 5,
        reliability_beta: float = 1.0,
        mad_epsilon: float = 1e-6,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.adversarial_weight = float(adversarial_weight)
        self.gradient_clip = float(gradient_clip)
        self.use_adaptive_weight = bool(use_adaptive_weight)
        self.use_gan = bool(use_gan)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.reliability_beta = float(reliability_beta)
        self.mad_epsilon = float(mad_epsilon)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.history: List[Dict[str, float]] = []
        self.calibration_median: Optional[float] = None
        self.calibration_scale: Optional[float] = None

    def _alpha(self, epoch: int) -> float:
        if not self.use_adaptive_weight or epoch <= self.warmup_epochs:
            return 0.0
        if epoch < self.warmup_epochs + self.ramp_epochs:
            return float(epoch - self.warmup_epochs) / float(self.ramp_epochs)
        return 1.0

    def _weights(self, sample_errors: torch.Tensor, alpha: float) -> torch.Tensor:
        if not self.use_adaptive_weight:
            return torch.ones_like(sample_errors)
        return reliability_weights(
            sample_errors,
            self.calibration_median,
            self.calibration_scale,
            alpha,
            self.reliability_beta,
        )

    def pretrain_domain_autoencoder(self, loader: DataLoader, epochs: int) -> None:
        if epochs <= 0:
            return
        optimizer = torch.optim.Adam(self.generator.domain_autoencoder.parameters(), lr=1e-3)
        self.generator.train()
        for _ in range(epochs):
            for real in loader:
                real = real.to(self.device)
                reconstructed_patches, patches = self.generator.autoencode_patches(real)
                loss = torch.square(reconstructed_patches - patches).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _freeze_domain_modules(self) -> None:
        self.generator.freeze_domain_modules()
        trainable = [parameter for parameter in self.generator.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("generator has no trainable Stage-II parameters")
        self.generator_optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _validation_loss(self, loader: DataLoader, max_batches: int = 32) -> float:
        self.generator.eval()
        losses = []
        with torch.no_grad():
            for index, real in enumerate(loader):
                if index >= max_batches:
                    break
                real = real.to(self.device)
                reconstructed, _, _ = self.generator(real)
                losses.extend(sample_reconstruction_errors(real, reconstructed).cpu().tolist())
        return float(np.mean(losses)) if losses else float("inf")

    def calibrate(self, loader: DataLoader) -> Tuple[float, float]:
        """Freeze final training-only m_train and s_train for inference."""
        self.generator.eval()
        errors = []
        with torch.no_grad():
            for real in loader:
                real = real.to(self.device)
                reconstructed, _, _ = self.generator(real)
                errors.append(sample_reconstruction_errors(real, reconstructed).cpu())
        if not errors:
            raise ValueError("cannot calibrate on an empty training loader")
        self.calibration_median, self.calibration_scale = robust_error_statistics(
            torch.cat(errors), self.mad_epsilon
        )
        return self.calibration_median, self.calibration_scale

    def _training_errors(self, loader: DataLoader) -> torch.Tensor:
        """Collect complete-dataset errors after an epoch with fixed parameters."""
        self.generator.eval()
        errors = []
        with torch.no_grad():
            for real in loader:
                real = real.to(self.device)
                reconstructed, _, _ = self.generator(real)
                errors.append(sample_reconstruction_errors(real, reconstructed).cpu())
        if not errors:
            raise ValueError("cannot estimate training statistics from an empty loader")
        return torch.cat(errors)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 80,
        early_stop_patience: int = 10,
        checkpoint_path: Path = None,
    ) -> List[Dict[str, float]]:
        self._freeze_domain_modules()
        best_loss = float("inf")
        best_state = None
        stale = 0
        for epoch_index in range(epochs):
            epoch = epoch_index + 1
            alpha = self._alpha(epoch)
            self.generator.train()
            self.generator.freeze_domain_modules()
            self.discriminator.train()
            generator_losses, discriminator_losses = [], []
            reconstruction_losses, observed_weights = [], []
            for real in train_loader:
                real = real.to(self.device)

                if self.use_gan:
                    with torch.no_grad():
                        reconstructed, encoded_latent, reconstructed_latent = self.generator(real)
                        sample_errors = sample_reconstruction_errors(real, reconstructed)
                        weights = self._weights(sample_errors, alpha)
                    real_logits = self.discriminator(encoded_latent.detach())
                    fake_logits = self.discriminator(reconstructed_latent.detach())
                    real_loss = self.bce(real_logits, torch.ones_like(real_logits)).mean(dim=1)
                    fake_loss = self.bce(fake_logits, torch.zeros_like(fake_logits)).mean(dim=1)
                    discriminator_loss = weighted_mean(
                        real_loss + fake_loss, weights, self.mad_epsilon
                    )
                    self.discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
                    self.discriminator_optimizer.step()
                else:
                    discriminator_loss = torch.zeros((), device=self.device)

                reconstructed, _, reconstructed_latent = self.generator(real)
                sample_errors = sample_reconstruction_errors(real, reconstructed)
                weights = self._weights(sample_errors.detach(), alpha)
                reconstruction_loss = weighted_mean(
                    sample_errors, weights, self.mad_epsilon
                )
                if self.use_gan:
                    set_requires_grad(self.discriminator, False)
                    fake_logits = self.discriminator(reconstructed_latent)
                    adversarial_per_sample = self.bce(
                        fake_logits, torch.ones_like(fake_logits)
                    ).mean(dim=1)
                    adversarial_loss = weighted_mean(
                        adversarial_per_sample, weights, self.mad_epsilon
                    )
                else:
                    adversarial_loss = torch.zeros((), device=self.device)
                generator_loss = reconstruction_loss + self.adversarial_weight * adversarial_loss
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
                self.generator_optimizer.step()
                if self.use_gan:
                    set_requires_grad(self.discriminator, True)

                generator_losses.append(float(generator_loss.detach().cpu()))
                discriminator_losses.append(float(discriminator_loss.detach().cpu()))
                reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
                observed_weights.append(weights.detach().cpu())

            # Method: statistics are computed from the complete training set
            # after epoch e and become weights only in epoch e+1.
            epoch_errors = self._training_errors(train_loader)
            next_median, next_scale = robust_error_statistics(epoch_errors, self.mad_epsilon)
            validation_loss = self._validation_loss(val_loader)
            row = {
                "epoch": float(epoch),
                "generator_loss": float(np.mean(generator_losses)),
                "discriminator_loss": float(np.mean(discriminator_losses)),
                "reconstruction_loss": float(np.mean(reconstruction_losses)),
                "validation_loss": validation_loss,
                "weight_alpha": float(alpha),
                "mean_sample_weight": float(torch.cat(observed_weights).mean()),
                "calibration_median": float(next_median),
                "calibration_scale": float(next_scale),
            }
            self.history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            # Statistics from epoch e become available only for epoch e+1.
            self.calibration_median, self.calibration_scale = next_median, next_scale

            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.generator.state_dict())
                stale = 0
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, str(checkpoint_path))
            else:
                stale += 1
                if early_stop_patience > 0 and stale > early_stop_patience:
                    break
        if best_state is not None:
            self.generator.load_state_dict(best_state)
        # Keep calibration consistent with the selected final generator and
        # use training observations only.
        self.calibrate(train_loader)
        return self.history


class MultiDomainHTADTrainer:
    """Two-stage trainer with domain-specific statistics and shared G/D."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        adversarial_weight: float = 0.01,
        weight_decay: float = 0.0,
        gradient_clip: float = 5.0,
        use_adaptive_weight: bool = True,
        use_gan: bool = True,
        warmup_epochs: int = 3,
        ramp_epochs: int = 5,
        reliability_beta: float = 1.0,
        mad_epsilon: float = 1e-6,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adversarial_weight = float(adversarial_weight)
        self.gradient_clip = float(gradient_clip)
        self.use_adaptive_weight = bool(use_adaptive_weight)
        self.use_gan = bool(use_gan)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.reliability_beta = float(reliability_beta)
        self.mad_epsilon = float(mad_epsilon)
        self.generator_optimizer = None
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.history: List[Dict[str, object]] = []
        self.calibration: Dict[str, Tuple[float, float]] = {}

    def _alpha(self, epoch: int) -> float:
        if not self.use_adaptive_weight or epoch <= self.warmup_epochs:
            return 0.0
        if epoch < self.warmup_epochs + self.ramp_epochs:
            return float(epoch - self.warmup_epochs) / float(self.ramp_epochs)
        return 1.0

    def _weights(self, domain: str, errors: torch.Tensor, alpha: float) -> torch.Tensor:
        if not self.use_adaptive_weight or domain not in self.calibration:
            return torch.ones_like(errors)
        median, scale = self.calibration[domain]
        return reliability_weights(errors, median, scale, alpha, self.reliability_beta)

    def pretrain_domain_autoencoders(
        self, loaders: Mapping[str, DataLoader], epochs: int, learning_rate: float = 1e-3
    ) -> None:
        if epochs <= 0:
            return
        for domain, loader in loaders.items():
            module = self.generator.domain_autoencoders[domain]
            set_requires_grad(module, True)
            module.train()
            optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
            for _ in range(epochs):
                for real in loader:
                    real = real.to(self.device)
                    reconstructed_patches, patches = self.generator.autoencode_patches(real, domain)
                    loss = torch.square(reconstructed_patches - patches).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def _freeze_domain_modules(self) -> None:
        self.generator.freeze_domain_modules()
        trainable = [parameter for parameter in self.generator.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("generator has no trainable shared Stage-II parameters")
        self.generator_optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _domain_errors(self, domain: str, loader: DataLoader) -> torch.Tensor:
        self.generator.eval()
        errors = []
        with torch.no_grad():
            for real in loader:
                real = real.to(self.device)
                reconstructed, _, _ = self.generator(real, domain)
                errors.append(sample_reconstruction_errors(real, reconstructed).cpu())
        if not errors:
            raise ValueError("cannot estimate statistics for empty domain {}".format(domain))
        return torch.cat(errors)

    def _validation_losses(self, loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        losses = {}
        for domain, loader in loaders.items():
            values = self._domain_errors(domain, loader)
            losses[domain] = float(values.mean())
        return losses

    def calibrate(self, loaders: Mapping[str, DataLoader]) -> Dict[str, Tuple[float, float]]:
        calibration = {}
        for domain, loader in loaders.items():
            calibration[domain] = robust_error_statistics(
                self._domain_errors(domain, loader), self.mad_epsilon
            )
        self.calibration = calibration
        return dict(calibration)

    def fit(
        self,
        train_loaders: Mapping[str, DataLoader],
        val_loaders: Mapping[str, DataLoader],
        epochs: int = 80,
        early_stop_patience: int = 10,
        checkpoint_path: Path = None,
    ) -> List[Dict[str, object]]:
        if set(train_loaders) != set(val_loaders):
            raise ValueError("training and validation domains must match")
        self._freeze_domain_modules()
        best_loss = float("inf")
        best_state = None
        stale = 0

        for epoch_index in range(epochs):
            epoch = epoch_index + 1
            alpha = self._alpha(epoch)
            self.generator.train()
            self.generator.freeze_domain_modules()
            self.discriminator.train()
            generator_losses, discriminator_losses = [], []
            reconstruction_losses, observed_weights = [], []

            for domain, loader in train_loaders.items():
                for real in loader:
                    real = real.to(self.device)
                    if self.use_gan:
                        with torch.no_grad():
                            reconstructed, encoded_latent, reconstructed_latent = self.generator(
                                real, domain
                            )
                            sample_errors = sample_reconstruction_errors(real, reconstructed)
                            weights = self._weights(domain, sample_errors, alpha)
                        real_logits = self.discriminator(encoded_latent.detach())
                        fake_logits = self.discriminator(reconstructed_latent.detach())
                        real_loss = self.bce(real_logits, torch.ones_like(real_logits)).mean(dim=1)
                        fake_loss = self.bce(fake_logits, torch.zeros_like(fake_logits)).mean(dim=1)
                        discriminator_loss = weighted_mean(
                            real_loss + fake_loss, weights, self.mad_epsilon
                        )
                        self.discriminator_optimizer.zero_grad()
                        discriminator_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.gradient_clip
                        )
                        self.discriminator_optimizer.step()
                    else:
                        discriminator_loss = torch.zeros((), device=self.device)

                    reconstructed, _, reconstructed_latent = self.generator(real, domain)
                    sample_errors = sample_reconstruction_errors(real, reconstructed)
                    weights = self._weights(domain, sample_errors.detach(), alpha)
                    reconstruction_loss = weighted_mean(
                        sample_errors, weights, self.mad_epsilon
                    )
                    if self.use_gan:
                        set_requires_grad(self.discriminator, False)
                        fake_logits = self.discriminator(reconstructed_latent)
                        adversarial_per_sample = self.bce(
                            fake_logits, torch.ones_like(fake_logits)
                        ).mean(dim=1)
                        adversarial_loss = weighted_mean(
                            adversarial_per_sample, weights, self.mad_epsilon
                        )
                    else:
                        adversarial_loss = torch.zeros((), device=self.device)
                    generator_loss = (
                        reconstruction_loss + self.adversarial_weight * adversarial_loss
                    )
                    self.generator_optimizer.zero_grad()
                    generator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.generator.parameters() if p.requires_grad],
                        self.gradient_clip,
                    )
                    self.generator_optimizer.step()
                    if self.use_gan:
                        set_requires_grad(self.discriminator, True)

                    generator_losses.append(float(generator_loss.detach().cpu()))
                    discriminator_losses.append(float(discriminator_loss.detach().cpu()))
                    reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
                    observed_weights.append(weights.detach().cpu())

            next_calibration = {
                domain: robust_error_statistics(
                    self._domain_errors(domain, loader), self.mad_epsilon
                )
                for domain, loader in train_loaders.items()
            }
            validation = self._validation_losses(val_loaders)
            validation_loss = float(np.mean(list(validation.values())))
            row = {
                "epoch": float(epoch),
                "generator_loss": float(np.mean(generator_losses)),
                "discriminator_loss": float(np.mean(discriminator_losses)),
                "reconstruction_loss": float(np.mean(reconstruction_losses)),
                "validation_loss": validation_loss,
                "validation_by_domain": validation,
                "weight_alpha": float(alpha),
                "mean_sample_weight": float(torch.cat(observed_weights).mean()),
                "calibration_by_domain": {
                    domain: {"median": value[0], "scale": value[1]}
                    for domain, value in next_calibration.items()
                },
            }
            self.history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            self.calibration = next_calibration

            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.generator.state_dict())
                stale = 0
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, str(checkpoint_path))
            else:
                stale += 1
                if early_stop_patience > 0 and stale > early_stop_patience:
                    break

        if best_state is not None:
            self.generator.load_state_dict(best_state)
        self.calibrate(train_loaders)
        return self.history


def _forward_generator(generator: nn.Module, tensor: torch.Tensor, domain: Optional[str]):
    if domain is None:
        return generator(tensor)
    return generator(tensor, domain)


def reconstruct_segments(
    generator: nn.Module,
    segments: Sequence[np.ndarray],
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    domain: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if stride > window_size:
        raise ValueError("evaluation stride cannot exceed window_size")
    generator.eval()
    all_values, all_reconstructed = [], []
    with torch.no_grad():
        for segment in segments:
            starts = evaluation_windows(len(segment), window_size, stride)
            if not starts:
                continue
            reconstructed_sum = np.zeros_like(segment, dtype=np.float64)
            counts = np.zeros((len(segment), 1), dtype=np.float64)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                windows = np.stack([segment[start : start + window_size] for start in batch_starts])
                tensor = torch.from_numpy(windows.astype(np.float32, copy=False)).to(device)
                predicted, _, _ = _forward_generator(generator, tensor, domain)
                predicted = predicted.cpu().numpy()
                for start, values in zip(batch_starts, predicted):
                    reconstructed_sum[start : start + window_size] += values
                    counts[start : start + window_size] += 1.0
            if np.any(counts == 0):
                raise RuntimeError("evaluation windows did not cover the full segment")
            all_values.append(np.asarray(segment, dtype=np.float32))
            all_reconstructed.append((reconstructed_sum / counts).astype(np.float32))
    if not all_values:
        raise ValueError("no segment is long enough for the requested evaluation window")
    return np.concatenate(all_values), np.concatenate(all_reconstructed)


def method_anomaly_scores(
    generator: nn.Module,
    segments: Sequence[np.ndarray],
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    calibration_median: float,
    calibration_scale: float,
    domain: Optional[str] = None,
) -> np.ndarray:
    """Compute [ (R_test - m_train) / s_train ]_+ and map windows to points."""
    if stride > window_size:
        raise ValueError("evaluation stride cannot exceed window_size")
    generator.eval()
    all_scores = []
    scale = max(float(calibration_scale), 1e-12)
    with torch.no_grad():
        for segment in segments:
            starts = evaluation_windows(len(segment), window_size, stride)
            if not starts:
                continue
            score_sum = np.zeros(len(segment), dtype=np.float64)
            counts = np.zeros(len(segment), dtype=np.float64)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                windows = np.stack([segment[start : start + window_size] for start in batch_starts])
                tensor = torch.from_numpy(windows.astype(np.float32, copy=False)).to(device)
                reconstructed, _, _ = _forward_generator(generator, tensor, domain)
                errors = sample_reconstruction_errors(tensor, reconstructed)
                scores = torch.clamp((errors - float(calibration_median)) / scale, min=0.0)
                for start, score in zip(batch_starts, scores.cpu().numpy()):
                    score_sum[start : start + window_size] += float(score)
                    counts[start : start + window_size] += 1.0
            if np.any(counts == 0):
                raise RuntimeError("evaluation windows did not cover the full segment")
            all_scores.append(score_sum / counts)
    if not all_scores:
        raise ValueError("no segment is long enough for the requested evaluation window")
    return np.concatenate(all_scores)


def calibration_scores(
    generator: nn.Module,
    val_segments: Sequence[np.ndarray],
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    values, reconstruction = reconstruct_segments(
        generator, val_segments, window_size, stride, batch_size, device
    )
    return anomaly_scores(values, reconstruction)
