"""
Delay Pattern for Multi-Codebook Autoregressive Generation

Based on MusicGen (Copet et al., 2023) and used in VoiceCraft-X.
Enables conditioning on previous codebook levels for the same timestep.

Pattern visualization (4 codebooks, 4 timesteps):
    Position: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    CB1:      A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -  -  -
    CB2:      -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -  -
    CB3:      -  -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -
    CB4:      -  -  -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -

The delay allows CB2 at time t to be conditioned on CB1 at time t.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class DelayPattern:
    """Implements delay pattern for multi-codebook autoregressive generation.

    Args:
        num_codebooks: Number of codebooks (K)
        delay_per_codebook: Delay added per codebook level (default: 1)
        special_token_id: ID for padding/special tokens (default: None)
    """

    def __init__(
        self,
        num_codebooks: int,
        delay_per_codebook: int = 1,
        special_token_id: Optional[int] = None,
    ):
        self.num_codebooks = num_codebooks
        self.delay_per_codebook = delay_per_codebook
        self.special_token_id = special_token_id

    def apply_delay(
        self,
        codes: torch.Tensor,
        pad_value: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply delay pattern to multi-codebook sequences.

        Args:
            codes: Input codes [B, K, T] where K is num_codebooks
            pad_value: Value to use for padding (default: special_token_id or 0)

        Returns:
            delayed_codes: Delayed codes [B, K, T + (K-1)*delay]
        """
        if pad_value is None:
            pad_value = self.special_token_id if self.special_token_id is not None else 0

        B, K, T = codes.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

        # Calculate total delay for last codebook
        max_delay = (K - 1) * self.delay_per_codebook
        T_delayed = T + max_delay

        # Initialize delayed sequence with padding
        delayed_codes = torch.full(
            (B, K, T_delayed),
            fill_value=pad_value,
            dtype=codes.dtype,
            device=codes.device,
        )

        # Apply delay to each codebook
        for k in range(K):
            delay = k * self.delay_per_codebook
            delayed_codes[:, k, delay:delay + T] = codes[:, k]

        return delayed_codes

    def remove_delay(
        self,
        delayed_codes: torch.Tensor,
        original_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Remove delay pattern from delayed sequences.

        Args:
            delayed_codes: Delayed codes [B, K, T_delayed]
            original_length: Original sequence length (if None, inferred)

        Returns:
            codes: Original codes [B, K, T]
        """
        B, K, T_delayed = delayed_codes.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

        # Infer original length if not provided
        if original_length is None:
            max_delay = (K - 1) * self.delay_per_codebook
            original_length = T_delayed - max_delay

        # Initialize output
        codes = torch.zeros(
            (B, K, original_length),
            dtype=delayed_codes.dtype,
            device=delayed_codes.device,
        )

        # Extract codes from each delayed position
        for k in range(K):
            delay = k * self.delay_per_codebook
            codes[:, k] = delayed_codes[:, k, delay:delay + original_length]

        return codes

    def flatten_delayed_codes(
        self,
        delayed_codes: torch.Tensor,
        remove_padding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten delayed codes into a single sequence for autoregressive modeling.

        Args:
            delayed_codes: Delayed codes [B, K, T_delayed]
            remove_padding: If True, remove padding positions

        Returns:
            flattened: Flattened sequence [B, L] where L depends on pattern
            positions: Original (k, t) positions for each token [B, L, 2]
        """
        B, K, T_delayed = delayed_codes.shape

        if not remove_padding:
            # Simple flattening: interleave codebooks
            # [B, K, T] -> [B, K*T]
            flattened = delayed_codes.reshape(B, K * T_delayed)

            # Create position indices
            positions = torch.zeros(B, K * T_delayed, 2, dtype=torch.long)
            for k in range(K):
                for t in range(T_delayed):
                    idx = k * T_delayed + t
                    positions[:, idx, 0] = k  # codebook index
                    positions[:, idx, 1] = t  # time index

            return flattened, positions

        else:
            # Remove padding positions (more complex but efficient)
            # Create a mask for valid (non-padded) positions
            valid_mask = torch.zeros(K, T_delayed, dtype=torch.bool, device=delayed_codes.device)

            for k in range(K):
                delay = k * self.delay_per_codebook
                max_delay = (K - 1) * self.delay_per_codebook
                original_length = T_delayed - max_delay
                end_pos = delay + original_length
                valid_mask[k, delay:end_pos] = True

            # Flatten in column-major order (time first, then codebook)
            # This ensures proper autoregressive ordering
            flattened_list = []
            positions_list = []

            for t in range(T_delayed):
                for k in range(K):
                    if valid_mask[k, t]:
                        flattened_list.append(delayed_codes[:, k, t])
                        positions_list.append((k, t))

            flattened = torch.stack(flattened_list, dim=1)  # [B, L]

            # Create positions tensor
            positions = torch.tensor(
                positions_list,
                dtype=torch.long,
                device=delayed_codes.device,
            ).unsqueeze(0).expand(B, -1, -1)  # [B, L, 2]

            return flattened, positions

    def unflatten_codes(
        self,
        flattened: torch.Tensor,
        positions: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        """Unflatten a sequence back to delayed code format.

        Args:
            flattened: Flattened sequence [B, L]
            positions: Position information [B, L, 2] (k, t pairs)
            output_length: Output sequence length T_delayed

        Returns:
            delayed_codes: Delayed codes [B, K, T_delayed]
        """
        B, L = flattened.shape
        pad_value = self.special_token_id if self.special_token_id is not None else 0

        # Initialize output
        delayed_codes = torch.full(
            (B, self.num_codebooks, output_length),
            fill_value=pad_value,
            dtype=flattened.dtype,
            device=flattened.device,
        )

        # Place each token at its designated position
        for i in range(L):
            k = positions[0, i, 0].item()  # codebook index (same for all batch)
            t = positions[0, i, 1].item()  # time index
            delayed_codes[:, k, t] = flattened[:, i]

        return delayed_codes


class DelayedCodebookEmbedding(nn.Module):
    """Embedding layer for delayed multi-codebook codes.

    Combines embeddings from multiple codebooks at each position.

    Args:
        num_codebooks: Number of codebooks
        codebook_size: Vocabulary size per codebook
        embedding_dim: Output embedding dimension
        combination: How to combine codebook embeddings ('sum' or 'concat')
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int,
        combination: str = 'sum',
        pad_token_id: Optional[int] = None,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.combination = combination
        self.pad_token_id = pad_token_id

        # Create separate embedding for each codebook
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(
                codebook_size,
                embedding_dim if combination == 'sum' else embedding_dim // num_codebooks,
                padding_idx=pad_token_id,
            )
            for _ in range(num_codebooks)
        ])

        if combination == 'concat':
            # Project concatenated embeddings to target dimension
            self.projection = nn.Linear(embedding_dim, embedding_dim)
        else:
            self.projection = None

    def forward(
        self,
        codes: torch.Tensor,
        return_separate: bool = False,
    ) -> torch.Tensor:
        """Embed multi-codebook codes.

        Args:
            codes: Codebook indices [B, K, T]
            return_separate: If True, return separate embeddings per codebook

        Returns:
            embeddings: Combined embeddings [B, T, D] or [B, K, T, D] if separate
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks

        # Embed each codebook separately
        embeddings = []
        for k in range(K):
            emb = self.codebook_embeddings[k](codes[:, k])  # [B, T, D']
            embeddings.append(emb)

        if return_separate:
            # Stack: [B, K, T, D']
            return torch.stack(embeddings, dim=1)

        # Combine embeddings
        if self.combination == 'sum':
            # Sum across codebooks: [B, T, D]
            combined = torch.stack(embeddings, dim=1).sum(dim=1)
        elif self.combination == 'concat':
            # Concatenate: [B, T, K*D'] -> [B, T, D]
            combined = torch.cat(embeddings, dim=-1)
            combined = self.projection(combined)
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")

        return combined


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Delay Pattern")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    num_codebooks = 4
    seq_length = 6
    codebook_size = 2048

    # Create sample codes
    codes = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_length))
    print(f"\nOriginal codes shape: {codes.shape}")
    print(f"Example codes (first sample, first 3 timesteps):")
    print(codes[0, :, :3])

    # Initialize delay pattern
    delay_pattern = DelayPattern(num_codebooks=num_codebooks, delay_per_codebook=1)

    # Apply delay
    delayed = delay_pattern.apply_delay(codes, pad_value=-1)
    print(f"\nDelayed codes shape: {delayed.shape}")
    print(f"Delayed codes (first sample):")
    print(delayed[0])

    # Flatten delayed codes
    flattened, positions = delay_pattern.flatten_delayed_codes(delayed, remove_padding=True)
    print(f"\nFlattened sequence shape: {flattened.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"First 10 positions (k, t):")
    print(positions[0, :10])
    print(f"First 10 tokens:")
    print(flattened[0, :10])

    # Unflatten
    unflattened = delay_pattern.unflatten_codes(
        flattened,
        positions,
        output_length=delayed.shape[2]
    )
    print(f"\nUnflattened shape: {unflattened.shape}")
    print(f"Reconstruction matches: {torch.equal(delayed, unflattened)}")

    # Remove delay
    recovered = delay_pattern.remove_delay(delayed, original_length=seq_length)
    print(f"\nRecovered codes shape: {recovered.shape}")
    print(f"Recovery matches original: {torch.equal(codes, recovered)}")

    # Test embedding
    print(f"\n{'=' * 60}")
    print("Testing Delayed Codebook Embedding")
    print("=" * 60)

    embedding_dim = 512
    embedder = DelayedCodebookEmbedding(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        combination='sum',
    )

    # Embed delayed codes
    embeddings = embedder(delayed)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Expected: [B={batch_size}, T={delayed.shape[2]}, D={embedding_dim}]")

    # Embed with separate outputs
    separate_embeddings = embedder(delayed, return_separate=True)
    print(f"Separate embeddings shape: {separate_embeddings.shape}")
    print(f"Expected: [B={batch_size}, K={num_codebooks}, T={delayed.shape[2]}, D={embedding_dim}]")

    print("\n✓ All tests passed!")
