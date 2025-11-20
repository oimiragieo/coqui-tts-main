"""
Weighted Loss Function for VoiceCraft-X

Based on VoiceCraft-X paper (arXiv:2511.12347v1).
Implements weighted cross-entropy loss with:
1. Codebook weighting: Different weights for each RVQ layer
2. Segment weighting: Higher weight for target (middle) segment

Loss formula:
    L = sum_i [ w_seg(z_i) * α_k * CE(pred_i, target_i) ]

where:
    - w_seg(z_i): Segment weight (1.0 for prefix/suffix, 3.0 for middle)
    - α_k: Codebook weight (1.0, 0.8, 0.6, 0.4 for codebooks 1-4)
    - CE: Cross-entropy loss
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceCraftXLoss(nn.Module):
    """Weighted loss for VoiceCraft-X training.

    Args:
        num_codebooks: Number of RVQ codebooks (default: 4)
        codebook_weights: Weights for each codebook (default: [1.0, 0.8, 0.6, 0.4])
        segment_weights: Dict of segment weights (default: {"prefix": 1.0, "suffix": 1.0, "middle": 3.0})
        ignore_index: Index to ignore in loss computation (default: -100)
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_weights: Optional[List[float]] = None,
        segment_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.ignore_index = ignore_index

        # Default codebook weights from paper
        if codebook_weights is None:
            codebook_weights = [1.0, 0.8, 0.6, 0.4][:num_codebooks]

        assert len(codebook_weights) == num_codebooks, \
            f"Expected {num_codebooks} codebook weights, got {len(codebook_weights)}"

        self.register_buffer(
            "codebook_weights",
            torch.tensor(codebook_weights, dtype=torch.float32)
        )

        # Default segment weights from paper
        if segment_weights is None:
            segment_weights = {
                "prefix": 1.0,
                "suffix": 1.0,
                "middle": 3.0,
            }

        self.segment_weights = segment_weights

    def compute_segment_mask(
        self,
        sequence_length: int,
        prefix_length: int,
        suffix_length: int,
        middle_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute segment weight mask.

        Args:
            sequence_length: Total sequence length
            prefix_length: Length of prefix segment
            suffix_length: Length of suffix segment
            middle_length: Length of middle (target) segment
            device: Device for tensor

        Returns:
            Segment weights [T] where each position has appropriate weight
        """
        segment_mask = torch.ones(sequence_length, dtype=torch.float32, device=device)

        # Assign weights to each segment
        # Assuming order: prefix + suffix + middle
        prefix_end = prefix_length
        suffix_end = prefix_end + suffix_length
        middle_end = suffix_end + middle_length

        # Prefix
        if prefix_length > 0:
            segment_mask[:prefix_end] = self.segment_weights["prefix"]

        # Suffix
        if suffix_length > 0:
            segment_mask[prefix_end:suffix_end] = self.segment_weights["suffix"]

        # Middle (target)
        if middle_length > 0:
            segment_mask[suffix_end:middle_end] = self.segment_weights["middle"]

        return segment_mask

    def forward(
        self,
        logits: List[torch.Tensor],
        targets: torch.Tensor,
        segment_lengths: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted cross-entropy loss.

        Args:
            logits: List of logits for each codebook, each [B, T, vocab_size]
            targets: Target token IDs [B, K, T] where K is num_codebooks
            segment_lengths: Optional dict with "prefix", "suffix", "middle" lengths

        Returns:
            loss: Total weighted loss
            loss_dict: Dictionary with detailed loss breakdown
        """
        assert len(logits) == self.num_codebooks, \
            f"Expected {self.num_codebooks} logit tensors, got {len(logits)}"

        B, K, T = targets.shape
        assert K == self.num_codebooks

        device = targets.device

        # Compute segment weights if lengths provided
        if segment_lengths is not None:
            segment_mask = self.compute_segment_mask(
                sequence_length=T,
                prefix_length=segment_lengths.get("prefix", 0),
                suffix_length=segment_lengths.get("suffix", 0),
                middle_length=segment_lengths.get("middle", 0),
                device=device,
            )
            # Expand to batch: [B, T]
            segment_mask = segment_mask.unsqueeze(0).expand(B, -1)
        else:
            # Uniform weighting if no segment info
            segment_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        # Compute loss for each codebook
        total_loss = 0.0
        codebook_losses = []

        for k in range(self.num_codebooks):
            # Get logits and targets for this codebook
            logits_k = logits[k]  # [B, T, vocab_size]
            targets_k = targets[:, k, :]  # [B, T]

            # Flatten for CE loss
            logits_k_flat = logits_k.reshape(-1, logits_k.size(-1))  # [B*T, vocab_size]
            targets_k_flat = targets_k.reshape(-1)  # [B*T]

            # Compute CE loss (unreduced)
            ce_loss = F.cross_entropy(
                logits_k_flat,
                targets_k_flat,
                ignore_index=self.ignore_index,
                reduction='none',
            )  # [B*T]

            # Reshape back to [B, T]
            ce_loss = ce_loss.reshape(B, T)

            # Apply segment weighting
            weighted_ce = ce_loss * segment_mask  # [B, T]

            # Reduce to scalar
            codebook_loss = weighted_ce.sum() / (segment_mask.sum() + 1e-8)

            # Apply codebook weight
            codebook_weight = self.codebook_weights[k]
            weighted_loss = codebook_weight * codebook_loss

            # Accumulate
            total_loss = total_loss + weighted_loss
            codebook_losses.append(codebook_loss.detach())

        # Prepare detailed loss dictionary
        loss_dict = {
            "total_loss": total_loss.detach(),
            "codebook_losses": torch.stack(codebook_losses),
        }

        # Add per-segment losses if segment info available
        if segment_lengths is not None:
            segment_losses = self._compute_segment_losses(
                logits, targets, segment_lengths
            )
            loss_dict.update(segment_losses)

        return total_loss, loss_dict

    def _compute_segment_losses(
        self,
        logits: List[torch.Tensor],
        targets: torch.Tensor,
        segment_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """Compute separate losses for each segment.

        Args:
            logits: List of logits for each codebook
            targets: Target token IDs [B, K, T]
            segment_lengths: Segment lengths dict

        Returns:
            Dictionary with prefix_loss, suffix_loss, middle_loss
        """
        B, K, T = targets.shape

        prefix_len = segment_lengths.get("prefix", 0)
        suffix_len = segment_lengths.get("suffix", 0)
        middle_len = segment_lengths.get("middle", 0)

        # Determine segment boundaries (assuming order: prefix + suffix + middle)
        prefix_end = prefix_len
        suffix_end = prefix_end + suffix_len
        middle_end = suffix_end + middle_len

        segment_losses = {}

        # Compute loss for each segment
        for segment_name, (start, end) in [
            ("prefix", (0, prefix_end)),
            ("suffix", (prefix_end, suffix_end)),
            ("middle", (suffix_end, middle_end)),
        ]:
            if end > start:
                segment_loss = 0.0

                for k in range(self.num_codebooks):
                    # Extract segment
                    logits_k_seg = logits[k][:, start:end, :]  # [B, T_seg, vocab]
                    targets_k_seg = targets[:, k, start:end]  # [B, T_seg]

                    # Compute CE loss
                    logits_flat = logits_k_seg.reshape(-1, logits_k_seg.size(-1))
                    targets_flat = targets_k_seg.reshape(-1)

                    ce_loss = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        ignore_index=self.ignore_index,
                        reduction='mean',
                    )

                    # Weight by codebook
                    weighted = self.codebook_weights[k] * ce_loss
                    segment_loss = segment_loss + weighted

                segment_losses[f"{segment_name}_loss"] = segment_loss.detach()
            else:
                segment_losses[f"{segment_name}_loss"] = torch.tensor(0.0)

        return segment_losses


class DelayedCodebookLoss(nn.Module):
    """Loss for delayed multi-codebook sequences.

    Handles loss computation for sequences with delay pattern.

    Args:
        num_codebooks: Number of codebooks
        codebook_weights: Weights for each codebook
        ignore_index: Index to ignore in loss
        delay_per_codebook: Delay steps per codebook (default: 1)
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        delay_per_codebook: int = 1,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.ignore_index = ignore_index
        self.delay_per_codebook = delay_per_codebook

        if codebook_weights is None:
            codebook_weights = [1.0, 0.8, 0.6, 0.4][:num_codebooks]

        self.register_buffer(
            "codebook_weights",
            torch.tensor(codebook_weights, dtype=torch.float32)
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss for delayed sequences.

        Args:
            logits: Flattened logits [B, L, vocab_size] where L is flattened length
            targets: Flattened target IDs [B, L]
            positions: Position info [B, L, 2] with (k, t) pairs

        Returns:
            loss: Total loss
            loss_dict: Detailed loss breakdown
        """
        B, L = targets.shape

        # Compute CE loss for all positions
        logits_flat = logits.reshape(-1, logits.size(-1))  # [B*L, vocab]
        targets_flat = targets.reshape(-1)  # [B*L]

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction='none',
        )  # [B*L]

        # Reshape to [B, L]
        ce_loss = ce_loss.reshape(B, L)

        # Apply codebook weights based on position info
        # positions: [B, L, 2] where positions[:, :, 0] is codebook index
        codebook_indices = positions[0, :, 0]  # [L] (same for all batch)

        # Create weight tensor
        weights = torch.zeros(L, device=targets.device)
        for k in range(self.num_codebooks):
            mask = (codebook_indices == k)
            weights[mask] = self.codebook_weights[k]

        # Apply weights [B, L]
        weighted_loss = ce_loss * weights.unsqueeze(0)

        # Reduce
        total_loss = weighted_loss.sum() / (weights.sum() + 1e-8)

        # Per-codebook losses
        codebook_losses = []
        for k in range(self.num_codebooks):
            mask = (codebook_indices == k)
            if mask.sum() > 0:
                cb_loss = ce_loss[:, mask].mean()
            else:
                cb_loss = torch.tensor(0.0, device=targets.device)
            codebook_losses.append(cb_loss)

        loss_dict = {
            "total_loss": total_loss.detach(),
            "codebook_losses": torch.stack(codebook_losses),
        }

        return total_loss, loss_dict


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing VoiceCraft-X Loss")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    num_codebooks = 4
    seq_length = 100
    vocab_size = 2048

    # Create sample data
    logits = [
        torch.randn(batch_size, seq_length, vocab_size)
        for _ in range(num_codebooks)
    ]
    targets = torch.randint(0, vocab_size, (batch_size, num_codebooks, seq_length))

    # Define segment lengths
    segment_lengths = {
        "prefix": 20,
        "suffix": 30,
        "middle": 50,
    }

    print(f"\nInput:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Num codebooks: {num_codebooks}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Segment lengths: {segment_lengths}")

    # Test loss computation
    loss_fn = VoiceCraftXLoss(
        num_codebooks=num_codebooks,
        codebook_weights=[1.0, 0.8, 0.6, 0.4],
        segment_weights={"prefix": 1.0, "suffix": 1.0, "middle": 3.0},
    )

    print(f"\n{'=' * 60}")
    print("Computing Loss with Segment Weighting")
    print("=" * 60)

    loss, loss_dict = loss_fn(logits, targets, segment_lengths=segment_lengths)

    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"\nPer-codebook losses:")
    for k, cb_loss in enumerate(loss_dict["codebook_losses"]):
        print(f"  Codebook {k}: {cb_loss.item():.4f}")

    print(f"\nPer-segment losses:")
    for key in ["prefix_loss", "suffix_loss", "middle_loss"]:
        if key in loss_dict:
            print(f"  {key}: {loss_dict[key].item():.4f}")

    # Test without segment weighting
    print(f"\n{'=' * 60}")
    print("Computing Loss without Segment Weighting")
    print("=" * 60)

    loss_uniform, loss_dict_uniform = loss_fn(logits, targets, segment_lengths=None)

    print(f"\nTotal loss: {loss_uniform.item():.4f}")
    print(f"Difference from weighted: {(loss - loss_uniform).item():.4f}")

    # Test delayed codebook loss
    print(f"\n{'=' * 60}")
    print("Testing Delayed Codebook Loss")
    print("=" * 60)

    # Create flattened sequence
    flattened_length = 200
    logits_flat = torch.randn(batch_size, flattened_length, vocab_size)
    targets_flat = torch.randint(0, vocab_size, (batch_size, flattened_length))

    # Create position info (random for testing)
    positions = torch.zeros(batch_size, flattened_length, 2, dtype=torch.long)
    for i in range(flattened_length):
        k = i % num_codebooks  # Codebook index
        t = i // num_codebooks  # Time index
        positions[:, i, 0] = k
        positions[:, i, 1] = t

    delayed_loss_fn = DelayedCodebookLoss(
        num_codebooks=num_codebooks,
        codebook_weights=[1.0, 0.8, 0.6, 0.4],
    )

    loss_delayed, loss_dict_delayed = delayed_loss_fn(
        logits_flat,
        targets_flat,
        positions,
    )

    print(f"\nTotal loss: {loss_delayed.item():.4f}")
    print(f"\nPer-codebook losses:")
    for k, cb_loss in enumerate(loss_dict_delayed["codebook_losses"]):
        print(f"  Codebook {k}: {cb_loss.item():.4f}")

    print("\n✓ All tests passed!")
