"""
EnCodec-style Speech Tokenizer with Residual Vector Quantization (RVQ)

Based on VoiceCraft-X paper (arXiv:2511.12347v1):
- 4 RVQ codebooks, each with 2048 entries
- 50Hz framerate (320 sample stride at 16kHz)
- Residual quantization for better compression
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization with multiple codebooks.

    Args:
        num_codebooks: Number of RVQ layers (default: 4)
        codebook_size: Number of entries per codebook (default: 2048)
        codebook_dim: Dimension of codebook vectors (default: 512)
        commitment_weight: Weight for commitment loss (default: 0.25)
        use_ema: Whether to use exponential moving average for codebook updates
        ema_decay: Decay rate for EMA (default: 0.99)
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 2048,
        codebook_dim: int = 512,
        commitment_weight: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Initialize codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim))
            for _ in range(num_codebooks)
        ])

        if use_ema:
            # EMA cluster sizes and embeddings
            self.register_buffer(
                "cluster_sizes",
                torch.zeros(num_codebooks, codebook_size)
            )
            self.register_buffer(
                "embed_avg",
                torch.zeros(num_codebooks, codebook_size, codebook_dim)
            )
            # Initialize EMA embeddings
            for i in range(num_codebooks):
                self.embed_avg[i].data.copy_(self.codebooks[i].data)
                self.cluster_sizes[i].fill_(1.0)

    def _initialize_codebook(self, x: torch.Tensor, codebook_idx: int):
        """Initialize codebook with random samples from input."""
        flat_x = rearrange(x, 'b d t -> (b t) d')

        # Random initialization
        if len(flat_x) >= self.codebook_size:
            indices = torch.randperm(len(flat_x))[:self.codebook_size]
            self.codebooks[codebook_idx].data.copy_(flat_x[indices])
        else:
            # Repeat if not enough samples
            n_repeat = math.ceil(self.codebook_size / len(flat_x))
            repeated = flat_x.repeat(n_repeat, 1)[:self.codebook_size]
            self.codebooks[codebook_idx].data.copy_(repeated)

    def _quantize_single(
        self,
        x: torch.Tensor,
        codebook_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize using a single codebook.

        Args:
            x: Input tensor [B, D, T]
            codebook_idx: Index of codebook to use

        Returns:
            quantized: Quantized tensor [B, D, T]
            indices: Codebook indices [B, T]
            commit_loss: Commitment loss
        """
        B, D, T = x.shape
        codebook = self.codebooks[codebook_idx]

        # Flatten to [B*T, D]
        flat_x = rearrange(x, 'b d t -> (b t) d')

        # Compute distances to all codebook entries
        # distances: [B*T, codebook_size]
        distances = torch.cdist(flat_x, codebook, p=2)

        # Get nearest codebook indices
        indices = torch.argmin(distances, dim=1)  # [B*T]

        # Lookup quantized values
        quantized_flat = codebook[indices]  # [B*T, D]
        quantized = rearrange(quantized_flat, '(b t) d -> b d t', b=B, t=T)

        # Reshape indices
        indices = rearrange(indices, '(b t) -> b t', b=B, t=T)

        # Compute commitment loss
        if self.training:
            commit_loss = F.mse_loss(quantized.detach(), x) * self.commitment_weight
        else:
            commit_loss = torch.tensor(0.0, device=x.device)

        # Update codebook with EMA if training
        if self.training and self.use_ema:
            self._update_ema(flat_x, indices.flatten(), codebook_idx)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, indices, commit_loss

    def _update_ema(self, x: torch.Tensor, indices: torch.Tensor, codebook_idx: int):
        """Update codebook using exponential moving average.

        Args:
            x: Input tensor [N, D]
            indices: Codebook indices [N]
            codebook_idx: Index of codebook to update
        """
        # One-hot encode indices
        encodings = F.one_hot(indices, self.codebook_size).float()  # [N, K]

        # Update cluster sizes
        updated_cluster_sizes = encodings.sum(0)  # [K]
        self.cluster_sizes[codebook_idx].data.mul_(self.ema_decay).add_(
            updated_cluster_sizes, alpha=1 - self.ema_decay
        )

        # Laplace smoothing
        n = self.cluster_sizes[codebook_idx].sum()
        cluster_sizes = (
            (self.cluster_sizes[codebook_idx] + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Update embeddings
        embed_sum = encodings.T @ x  # [K, D]
        self.embed_avg[codebook_idx].data.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Normalize embeddings by cluster size
        self.codebooks[codebook_idx].data.copy_(
            self.embed_avg[codebook_idx] / cluster_sizes.unsqueeze(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_codes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual vector quantization forward pass.

        Args:
            x: Input tensor [B, D, T]
            return_all_codes: If True, return codes from all layers

        Returns:
            quantized: Final quantized tensor [B, D, T]
            indices: Codebook indices [B, num_codebooks, T] or [B, T]
            loss: Total quantization loss (commitment + diversity)
        """
        residual = x
        quantized_out = torch.zeros_like(x)

        all_indices = []
        total_loss = 0.0

        for i in range(self.num_codebooks):
            # Quantize residual
            quantized, indices, commit_loss = self._quantize_single(residual, i)

            # Accumulate quantized output
            quantized_out = quantized_out + quantized

            # Compute new residual
            residual = residual - quantized

            # Accumulate losses and indices
            total_loss = total_loss + commit_loss
            all_indices.append(indices)

        # Stack indices: [B, num_codebooks, T]
        all_indices = torch.stack(all_indices, dim=1)

        if return_all_codes:
            return quantized_out, all_indices, total_loss
        else:
            # Return only first codebook indices for compatibility
            return quantized_out, all_indices[:, 0], total_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to continuous representation.

        Args:
            indices: Codebook indices [B, num_codebooks, T]

        Returns:
            Reconstructed tensor [B, D, T]
        """
        B, K, T = indices.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

        output = torch.zeros(B, self.codebook_dim, T, device=indices.device)

        for i in range(self.num_codebooks):
            # Lookup from codebook
            codebook_indices = indices[:, i]  # [B, T]
            codebook = self.codebooks[i]  # [codebook_size, D]

            # Gather embeddings
            flat_indices = rearrange(codebook_indices, 'b t -> (b t)')
            quantized_flat = codebook[flat_indices]  # [B*T, D]
            quantized = rearrange(quantized_flat, '(b t) d -> b d t', b=B, t=T)

            # Accumulate
            output = output + quantized

        return output


class EnCodecTokenizer(nn.Module):
    """EnCodec-style neural audio codec with RVQ.

    Based on VoiceCraft-X specifications:
    - Input: 16kHz audio
    - Output: 50Hz token sequence with 4 codebooks
    - Stride: 320 samples (20ms)

    Args:
        num_codebooks: Number of RVQ layers (default: 4)
        codebook_size: Entries per codebook (default: 2048)
        codebook_dim: Codebook vector dimension (default: 512)
        hidden_dim: Encoder/decoder hidden dimension (default: 512)
        sample_rate: Audio sample rate (default: 16000)
        target_framerate: Target token framerate in Hz (default: 50)
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 2048,
        codebook_dim: int = 512,
        hidden_dim: int = 512,
        sample_rate: int = 16000,
        target_framerate: int = 50,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.target_framerate = target_framerate

        # Calculate hop length for target framerate
        self.hop_length = sample_rate // target_framerate  # 16000 / 50 = 320

        # Encoder: Strided convolutions to downsample
        # Target: 320x compression (stride 320)
        # Architecture: 3 conv layers with strides (2, 8, 20) = 320
        self.encoder = nn.Sequential(
            # Conv1: stride 2
            nn.Conv1d(1, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim // 4),

            # Conv2: stride 8
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=7, stride=8, padding=3),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim // 2),

            # Conv3: stride 20 (total = 2*8*20 = 320)
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, stride=20, padding=3),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim),

            # Project to codebook dimension
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=3, stride=1, padding=1),
        )

        # Residual Vector Quantizer
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        # Decoder: Transposed convolutions to upsample
        self.decoder = nn.Sequential(
            # Project from codebook dimension
            nn.Conv1d(codebook_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim),

            # ConvTranspose1: stride 20
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim // 2,
                kernel_size=40, stride=20, padding=10, output_padding=0
            ),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim // 2),

            # ConvTranspose2: stride 8
            nn.ConvTranspose1d(
                hidden_dim // 2, hidden_dim // 4,
                kernel_size=16, stride=8, padding=4, output_padding=0
            ),
            nn.ELU(),
            nn.GroupNorm(8, hidden_dim // 4),

            # ConvTranspose3: stride 2 (total = 20*8*2 = 320)
            nn.ConvTranspose1d(
                hidden_dim // 4, 1,
                kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def encode(
        self,
        audio: torch.Tensor,
        return_all_codes: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to discrete tokens.

        Args:
            audio: Input audio [B, 1, T_audio] or [B, T_audio]
            return_all_codes: Return all codebook indices

        Returns:
            indices: Codebook indices [B, num_codebooks, T_tokens]
            loss: Quantization loss
        """
        # Handle input shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Encode to continuous representation
        z = self.encoder(audio)  # [B, codebook_dim, T_tokens]

        # Quantize
        quantized, indices, loss = self.quantizer(z, return_all_codes=return_all_codes)

        return indices, loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to audio.

        Args:
            indices: Codebook indices [B, num_codebooks, T_tokens]

        Returns:
            audio: Reconstructed audio [B, 1, T_audio]
        """
        # Decode indices to continuous representation
        z = self.quantizer.decode(indices)  # [B, codebook_dim, T_tokens]

        # Decode to audio
        audio = self.decoder(z)  # [B, 1, T_audio]

        return audio

    def forward(
        self,
        audio: torch.Tensor,
        return_all_codes: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-decode pass.

        Args:
            audio: Input audio [B, 1, T_audio] or [B, T_audio]
            return_all_codes: Return all codebook indices

        Returns:
            audio_recon: Reconstructed audio [B, 1, T_audio]
            indices: Codebook indices [B, num_codebooks, T_tokens]
            loss: Quantization loss
        """
        indices, quant_loss = self.encode(audio, return_all_codes=return_all_codes)
        audio_recon = self.decode(indices)

        # Match input length exactly
        if audio.dim() == 2:
            target_length = audio.shape[1]
        else:
            target_length = audio.shape[2]

        audio_recon = audio_recon[..., :target_length]

        return audio_recon, indices, quant_loss


# Example usage and testing
if __name__ == "__main__":
    # Test EnCodec tokenizer
    batch_size = 2
    duration = 3.0  # seconds
    sample_rate = 16000
    audio_length = int(duration * sample_rate)

    # Create random audio
    audio = torch.randn(batch_size, audio_length)

    # Initialize tokenizer
    tokenizer = EnCodecTokenizer(
        num_codebooks=4,
        codebook_size=2048,
        codebook_dim=512,
        hidden_dim=512,
        sample_rate=16000,
        target_framerate=50,
    )

    print(f"Input audio shape: {audio.shape}")

    # Encode
    indices, quant_loss = tokenizer.encode(audio)
    print(f"Encoded indices shape: {indices.shape}")
    print(f"Expected: [B={batch_size}, K=4, T={duration*50}]")
    print(f"Quantization loss: {quant_loss.item():.4f}")

    # Decode
    audio_recon = tokenizer.decode(indices)
    print(f"Reconstructed audio shape: {audio_recon.shape}")

    # Full forward pass
    audio_recon2, indices2, loss2 = tokenizer(audio)
    print(f"\nFull forward pass:")
    print(f"  Reconstructed audio: {audio_recon2.shape}")
    print(f"  Indices: {indices2.shape}")
    print(f"  Loss: {loss2.item():.4f}")

    # Check reconstruction error
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    recon_error = F.mse_loss(audio_recon2, audio)
    print(f"  Reconstruction MSE: {recon_error.item():.4f}")
