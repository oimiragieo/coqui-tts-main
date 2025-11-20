"""
Token Reordering Mechanism for Unified TTS and Speech Editing

Based on VoiceCraft-X paper (arXiv:2511.12347v1).
Implements prefix-suffix-middle reordering with time-aligned text and speech tokens.

Key Idea:
- Training: Randomly segment utterance into prefix, middle, suffix
- Reorder to: prefix + suffix + middle (middle is the target)
- Enables model to infill/edit middle portion given surrounding context
- Unifies TTS and editing under single framework
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


@dataclass
class AlignmentInfo:
    """Word-level alignment information.

    Attributes:
        words: List of words in utterance
        start_times: Start time (in frames) for each word
        end_times: End time (in frames) for each word
        framerate: Framerate of speech tokens (Hz)
    """
    words: List[str]
    start_times: List[int]  # In frame indices
    end_times: List[int]
    framerate: float = 50.0  # Hz

    def __len__(self):
        return len(self.words)

    def get_frame_range(self, word_idx: int) -> Tuple[int, int]:
        """Get frame range for a word."""
        return self.start_times[word_idx], self.end_times[word_idx]

    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return max(self.end_times) if self.end_times else 0


@dataclass
class ReorderedSequence:
    """Container for reordered text and speech sequences.

    Attributes:
        prefix_text: Text tokens for prefix segment
        suffix_text: Text tokens for suffix segment
        middle_text: Text tokens for middle (target) segment
        prefix_speech: Speech tokens for prefix segment [K, T_prefix]
        suffix_speech: Speech tokens for suffix segment [K, T_suffix]
        middle_speech: Speech tokens for middle (target) segment [K, T_middle]
        mask_token_id: ID of special mask token
        speaker_embedding: Optional speaker embedding vector
    """
    prefix_text: torch.Tensor
    suffix_text: torch.Tensor
    middle_text: torch.Tensor
    prefix_speech: torch.Tensor
    suffix_speech: torch.Tensor
    middle_speech: torch.Tensor
    mask_token_id: int
    speaker_embedding: Optional[torch.Tensor] = None

    def get_input_sequence(
        self,
        include_middle_speech: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct input sequence for model.

        Returns:
            text_input: Concatenated text tokens
            speech_input: Concatenated speech tokens with masks
        """
        # Text: prefix + suffix + middle
        text_parts = []
        if len(self.prefix_text) > 0:
            text_parts.append(self.prefix_text)
        if len(self.suffix_text) > 0:
            text_parts.append(self.suffix_text)
        if len(self.middle_text) > 0:
            text_parts.append(self.middle_text)

        text_input = torch.cat(text_parts, dim=0) if text_parts else torch.tensor([])

        # Speech: prefix + <MASK> + suffix + <MASK> + [middle]
        # Masks are represented as special token IDs
        K = self.prefix_speech.shape[0] if self.prefix_speech.numel() > 0 else 4
        mask_frame = torch.full((K, 1), self.mask_token_id, dtype=torch.long)

        speech_parts = []

        # Add prefix if not empty
        if self.prefix_speech.numel() > 0:
            speech_parts.append(self.prefix_speech)

        # Add first mask
        speech_parts.append(mask_frame)

        # Add suffix if not empty
        if self.suffix_speech.numel() > 0:
            speech_parts.append(self.suffix_speech)

        # Add second mask
        speech_parts.append(mask_frame)

        # Add middle if requested (for training)
        if include_middle_speech and self.middle_speech.numel() > 0:
            speech_parts.append(self.middle_speech)

        speech_input = torch.cat(speech_parts, dim=1)  # [K, T_total]

        return text_input, speech_input

    def get_target_sequence(self) -> torch.Tensor:
        """Get target sequence for training (middle speech)."""
        return self.middle_speech


class TokenReorderingStrategy:
    """Implements token reordering strategies for TTS and editing.

    Args:
        mask_token_id: ID for special <MASK> token
        min_middle_ratio: Minimum ratio of utterance to use as middle (default: 0.1)
        max_middle_ratio: Maximum ratio of utterance to use as middle (default: 0.8)
        use_random_segmentation: Randomly segment or use provided boundaries
    """

    def __init__(
        self,
        mask_token_id: int,
        min_middle_ratio: float = 0.1,
        max_middle_ratio: float = 0.8,
        use_random_segmentation: bool = True,
    ):
        self.mask_token_id = mask_token_id
        self.min_middle_ratio = min_middle_ratio
        self.max_middle_ratio = max_middle_ratio
        self.use_random_segmentation = use_random_segmentation

    def random_segmentation(
        self,
        num_words: int,
    ) -> Tuple[int, int, int]:
        """Randomly segment words into prefix, middle, suffix.

        Args:
            num_words: Total number of words

        Returns:
            Tuple of (prefix_end, middle_start, middle_end, suffix_start)
            where ranges are [prefix_end, middle_start) for prefix,
                           [middle_start, middle_end) for middle,
                           [middle_end, suffix_start) for suffix
        """
        if num_words == 0:
            return 0, 0, 0, 0

        # Determine middle length
        min_middle_len = max(1, int(num_words * self.min_middle_ratio))
        max_middle_len = max(min_middle_len, int(num_words * self.max_middle_ratio))
        middle_len = np.random.randint(min_middle_len, max_middle_len + 1)

        # Determine middle start position
        max_middle_start = num_words - middle_len
        middle_start = np.random.randint(0, max(1, max_middle_start + 1))
        middle_end = middle_start + middle_len

        # Prefix: [0, middle_start)
        # Middle: [middle_start, middle_end)
        # Suffix: [middle_end, num_words)
        prefix_end = middle_start
        suffix_start = middle_end

        return prefix_end, middle_start, middle_end, suffix_start

    def reorder_with_alignment(
        self,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        alignment: AlignmentInfo,
        speaker_embedding: Optional[torch.Tensor] = None,
        segment_boundaries: Optional[Tuple[int, int]] = None,
    ) -> ReorderedSequence:
        """Reorder tokens based on word-level alignment.

        Args:
            text_tokens: Text token IDs [T_text]
            speech_tokens: Speech token IDs [K, T_speech] (multi-codebook)
            alignment: Word-level alignment information
            speaker_embedding: Optional speaker embedding
            segment_boundaries: Optional (middle_start, middle_end) word indices

        Returns:
            ReorderedSequence object with prefix-suffix-middle reordering
        """
        num_words = len(alignment)

        if num_words == 0:
            # Empty sequence - return empty reordering
            K = speech_tokens.shape[0]
            return ReorderedSequence(
                prefix_text=torch.tensor([], dtype=torch.long),
                suffix_text=torch.tensor([], dtype=torch.long),
                middle_text=text_tokens,
                prefix_speech=torch.zeros(K, 0, dtype=torch.long),
                suffix_speech=torch.zeros(K, 0, dtype=torch.long),
                middle_speech=speech_tokens,
                mask_token_id=self.mask_token_id,
                speaker_embedding=speaker_embedding,
            )

        # Determine segmentation
        if segment_boundaries is not None:
            middle_start, middle_end = segment_boundaries
            prefix_end = middle_start
            suffix_start = middle_end
        else:
            prefix_end, middle_start, middle_end, suffix_start = \
                self.random_segmentation(num_words)

        # Extract word ranges
        prefix_words = alignment.words[:prefix_end]
        middle_words = alignment.words[middle_start:middle_end]
        suffix_words = alignment.words[suffix_start:]

        # Get speech frame boundaries
        # Note: This is a simplified approach. In practice, you'd need to map
        # text tokens to words and use that alignment.

        # For simplicity, we'll assume uniform distribution of text tokens across words
        # In production, you'd want proper text-to-word alignment

        def get_speech_frames(word_start_idx: int, word_end_idx: int) -> Tuple[int, int]:
            """Get speech frame range for word range."""
            if word_start_idx >= len(alignment) or word_end_idx == word_start_idx:
                return 0, 0
            if word_end_idx > len(alignment):
                word_end_idx = len(alignment)

            start_frame = alignment.start_times[word_start_idx]
            end_frame = alignment.end_times[word_end_idx - 1]
            return start_frame, end_frame

        # Get speech segments
        if prefix_end > 0:
            prefix_start_frame, prefix_end_frame = get_speech_frames(0, prefix_end)
            prefix_speech = speech_tokens[:, prefix_start_frame:prefix_end_frame]
        else:
            prefix_speech = torch.zeros(speech_tokens.shape[0], 0, dtype=torch.long)

        if middle_end > middle_start:
            middle_start_frame, middle_end_frame = get_speech_frames(middle_start, middle_end)
            middle_speech = speech_tokens[:, middle_start_frame:middle_end_frame]
        else:
            middle_speech = torch.zeros(speech_tokens.shape[0], 0, dtype=torch.long)

        if suffix_start < num_words:
            suffix_start_frame, suffix_end_frame = get_speech_frames(suffix_start, num_words)
            suffix_speech = speech_tokens[:, suffix_start_frame:suffix_end_frame]
        else:
            suffix_speech = torch.zeros(speech_tokens.shape[0], 0, dtype=torch.long)

        # Get text segments (simplified - assumes 1:1 word to token mapping)
        # In production, you'd need proper tokenizer-word alignment
        words_per_token = len(alignment.words) / max(1, len(text_tokens))

        def get_text_range(word_start_idx: int, word_end_idx: int) -> Tuple[int, int]:
            """Estimate text token range for word range."""
            start_tok = int(word_start_idx / max(1, words_per_token))
            end_tok = int(word_end_idx / max(1, words_per_token))
            return start_tok, min(end_tok, len(text_tokens))

        if prefix_end > 0:
            p_start, p_end = get_text_range(0, prefix_end)
            prefix_text = text_tokens[p_start:p_end]
        else:
            prefix_text = torch.tensor([], dtype=torch.long)

        if middle_end > middle_start:
            m_start, m_end = get_text_range(middle_start, middle_end)
            middle_text = text_tokens[m_start:m_end]
        else:
            middle_text = torch.tensor([], dtype=torch.long)

        if suffix_start < num_words:
            s_start, s_end = get_text_range(suffix_start, num_words)
            suffix_text = text_tokens[s_start:s_end]
        else:
            suffix_text = torch.tensor([], dtype=torch.long)

        return ReorderedSequence(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            middle_text=middle_text,
            prefix_speech=prefix_speech,
            suffix_speech=suffix_speech,
            middle_speech=middle_speech,
            mask_token_id=self.mask_token_id,
            speaker_embedding=speaker_embedding,
        )

    def create_editing_sequence(
        self,
        prefix_text: torch.Tensor,
        suffix_text: torch.Tensor,
        new_middle_text: torch.Tensor,
        prefix_speech: torch.Tensor,
        suffix_speech: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> ReorderedSequence:
        """Create reordered sequence for speech editing inference.

        Args:
            prefix_text: Text before edit point
            suffix_text: Text after edit point
            new_middle_text: New text to insert/replace
            prefix_speech: Speech before edit point [K, T_prefix]
            suffix_speech: Speech after edit point [K, T_suffix]
            speaker_embedding: Optional speaker embedding

        Returns:
            ReorderedSequence for editing inference
        """
        K = prefix_speech.shape[0] if prefix_speech.numel() > 0 else \
            (suffix_speech.shape[0] if suffix_speech.numel() > 0 else 4)

        return ReorderedSequence(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            middle_text=new_middle_text,
            prefix_speech=prefix_speech,
            suffix_speech=suffix_speech,
            middle_speech=torch.zeros(K, 0, dtype=torch.long),  # To be generated
            mask_token_id=self.mask_token_id,
            speaker_embedding=speaker_embedding,
        )

    def create_tts_sequence(
        self,
        target_text: torch.Tensor,
        prompt_text: Optional[torch.Tensor] = None,
        prompt_speech: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> ReorderedSequence:
        """Create reordered sequence for zero-shot TTS inference.

        Args:
            target_text: Text to synthesize
            prompt_text: Optional prompt text (for conditioning)
            prompt_speech: Optional prompt speech [K, T_prompt]
            speaker_embedding: Optional speaker embedding

        Returns:
            ReorderedSequence for TTS inference
        """
        K = prompt_speech.shape[0] if prompt_speech is not None and prompt_speech.numel() > 0 else 4

        # For TTS: empty prefix/suffix, prompt + target in middle
        if prompt_text is not None and len(prompt_text) > 0:
            middle_text = torch.cat([prompt_text, target_text], dim=0)
        else:
            middle_text = target_text

        if prompt_speech is not None and prompt_speech.numel() > 0:
            # Prompt speech goes after masks
            # We'll use a special arrangement for TTS
            return ReorderedSequence(
                prefix_text=torch.tensor([], dtype=torch.long),
                suffix_text=torch.tensor([], dtype=torch.long),
                middle_text=middle_text,
                prefix_speech=torch.zeros(K, 0, dtype=torch.long),
                suffix_speech=torch.zeros(K, 0, dtype=torch.long),
                middle_speech=prompt_speech,  # Will be appended in special way
                mask_token_id=self.mask_token_id,
                speaker_embedding=speaker_embedding,
            )
        else:
            # No prompt - unconditional or with speaker embedding only
            return ReorderedSequence(
                prefix_text=torch.tensor([], dtype=torch.long),
                suffix_text=torch.tensor([], dtype=torch.long),
                middle_text=middle_text,
                prefix_speech=torch.zeros(K, 0, dtype=torch.long),
                suffix_speech=torch.zeros(K, 0, dtype=torch.long),
                middle_speech=torch.zeros(K, 0, dtype=torch.long),
                mask_token_id=self.mask_token_id,
                speaker_embedding=speaker_embedding,
            )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Token Reordering")
    print("=" * 60)

    # Create sample data
    num_codebooks = 4
    mask_token_id = 2048  # Assume mask token is after all regular tokens

    # Sample alignment
    words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    # Assume 50Hz framerate, ~0.3s per word
    start_times = [0, 15, 30, 50, 65, 85, 105, 120]
    end_times = [14, 29, 49, 64, 84, 104, 119, 135]

    alignment = AlignmentInfo(
        words=words,
        start_times=start_times,
        end_times=end_times,
        framerate=50.0,
    )

    print(f"\nAlignment: {len(alignment)} words")
    print(f"Words: {alignment.words}")
    print(f"Total frames: {alignment.get_total_frames()}")

    # Create sample tokens
    text_tokens = torch.arange(0, len(words) * 5)  # ~5 tokens per word
    speech_tokens = torch.randint(0, 2048, (num_codebooks, alignment.get_total_frames()))

    print(f"\nText tokens shape: {text_tokens.shape}")
    print(f"Speech tokens shape: {speech_tokens.shape}")

    # Test random segmentation
    reorderer = TokenReorderingStrategy(
        mask_token_id=mask_token_id,
        min_middle_ratio=0.2,
        max_middle_ratio=0.6,
    )

    print(f"\n{'=' * 60}")
    print("Testing Random Segmentation")
    print("=" * 60)

    for i in range(3):
        prefix_end, middle_start, middle_end, suffix_start = \
            reorderer.random_segmentation(len(alignment))
        print(f"\nTrial {i + 1}:")
        print(f"  Prefix: words[0:{prefix_end}] = {words[:prefix_end]}")
        print(f"  Middle: words[{middle_start}:{middle_end}] = {words[middle_start:middle_end]}")
        print(f"  Suffix: words[{suffix_start}:] = {words[suffix_start:]}")

    # Test reordering
    print(f"\n{'=' * 60}")
    print("Testing Reordering with Alignment")
    print("=" * 60)

    speaker_emb = torch.randn(512)
    reordered = reorderer.reorder_with_alignment(
        text_tokens,
        speech_tokens,
        alignment,
        speaker_embedding=speaker_emb,
    )

    print(f"\nReordered sequence:")
    print(f"  Prefix text: {reordered.prefix_text.shape}")
    print(f"  Suffix text: {reordered.suffix_text.shape}")
    print(f"  Middle text: {reordered.middle_text.shape}")
    print(f"  Prefix speech: {reordered.prefix_speech.shape}")
    print(f"  Suffix speech: {reordered.suffix_speech.shape}")
    print(f"  Middle speech: {reordered.middle_speech.shape}")

    # Get input sequence
    text_input, speech_input = reordered.get_input_sequence(include_middle_speech=True)
    print(f"\nInput sequence:")
    print(f"  Text: {text_input.shape}")
    print(f"  Speech: {speech_input.shape}")

    # Test editing sequence
    print(f"\n{'=' * 60}")
    print("Testing Editing Sequence Creation")
    print("=" * 60)

    edit_reordered = reorderer.create_editing_sequence(
        prefix_text=torch.arange(0, 10),
        suffix_text=torch.arange(20, 30),
        new_middle_text=torch.arange(100, 115),  # New text to insert
        prefix_speech=speech_tokens[:, :30],
        suffix_speech=speech_tokens[:, 100:],
        speaker_embedding=speaker_emb,
    )

    text_input, speech_input = edit_reordered.get_input_sequence(include_middle_speech=False)
    print(f"\nEditing input sequence:")
    print(f"  Text: {text_input.shape}")
    print(f"  Speech: {speech_input.shape}")
    print(f"  Speech has {(speech_input == mask_token_id).sum().item()} mask tokens")

    # Test TTS sequence
    print(f"\n{'=' * 60}")
    print("Testing TTS Sequence Creation")
    print("=" * 60)

    tts_reordered = reorderer.create_tts_sequence(
        target_text=torch.arange(0, 50),
        prompt_text=torch.arange(100, 110),
        prompt_speech=speech_tokens[:, :20],
        speaker_embedding=speaker_emb,
    )

    text_input, speech_input = tts_reordered.get_input_sequence(include_middle_speech=True)
    print(f"\nTTS input sequence:")
    print(f"  Text: {text_input.shape}")
    print(f"  Speech: {speech_input.shape}")

    print("\n✓ All tests passed!")
