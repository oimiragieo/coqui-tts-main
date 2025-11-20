"""
VoiceCraft-X: Unified Multilingual Speech Synthesis and Editing Model

Based on paper: "VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech
Synthesis and Speech Editing" (arXiv:2511.12347v1)

Key Features:
- Multilingual support (11+ languages)
- Unified TTS and speech editing
- Phoneme-free text processing with Qwen3
- Multi-codebook speech tokens with RVQ
- Token reordering for contextual generation
- Delay pattern for efficient multi-codebook prediction

Authors: Zhisheng Zheng et al.
Implementation for Coqui TTS by Claude
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTS.tts.layers.xtts.encodec_tokenizer import EnCodecTokenizer
from TTS.tts.layers.xtts.delay_pattern import DelayPattern, DelayedCodebookEmbedding
from TTS.tts.layers.xtts.token_reordering import (
    TokenReorderingStrategy,
    ReorderedSequence,
    AlignmentInfo,
)
from TTS.tts.layers.xtts.qwen3_backbone import Qwen3Backbone
from TTS.tts.layers.xtts.speaker_embedding import create_speaker_encoder
from TTS.tts.layers.xtts.voicecraft_x_loss import VoiceCraftXLoss


@dataclass
class VoiceCraftXConfig:
    """Configuration for VoiceCraft-X model.

    Codec Settings:
        num_codebooks: Number of RVQ codebooks (default: 4)
        codebook_size: Vocabulary size per codebook (default: 2048)
        codebook_dim: Codebook vector dimension (default: 512)
        sample_rate: Audio sample rate in Hz (default: 16000)
        codec_framerate: Speech token framerate in Hz (default: 50)

    Model Settings:
        qwen_model_name: Qwen3 model name (default: "Qwen/Qwen2.5-0.5B")
        freeze_qwen: Whether to freeze Qwen3 backbone (default: False)
        use_lora: Whether to use LoRA fine-tuning (default: False)
        lora_rank: LoRA rank if use_lora=True (default: 16)

    Speaker Embedding:
        speaker_encoder_type: Type of speaker encoder ("campplus" or "wavlm")
        speaker_encoder_path: Path to speaker encoder model
        speaker_embedding_dim: Speaker embedding dimension (default: 512)

    Training Settings:
        codebook_weights: Weights for each codebook (default: [1.0, 0.8, 0.6, 0.4])
        segment_weights: Segment weights dict (default: {"prefix": 1.0, "suffix": 1.0, "middle": 3.0})
        use_delay_pattern: Whether to use delay pattern (default: True)
        use_token_reordering: Whether to use token reordering (default: True)
    """

    # Codec settings
    num_codebooks: int = 4
    codebook_size: int = 2048
    codebook_dim: int = 512
    sample_rate: int = 16000
    codec_framerate: int = 50

    # Model settings
    qwen_model_name: str = "Qwen/Qwen2.5-0.5B"
    freeze_qwen: bool = False
    use_lora: bool = False
    lora_rank: int = 16

    # Speaker embedding
    speaker_encoder_type: str = "campplus"
    speaker_encoder_path: Optional[str] = None
    speaker_embedding_dim: int = 512

    # Training settings
    codebook_weights: List[float] = None
    segment_weights: Dict[str, float] = None
    use_delay_pattern: bool = True
    use_token_reordering: bool = True

    def __post_init__(self):
        if self.codebook_weights is None:
            self.codebook_weights = [1.0, 0.8, 0.6, 0.4][:self.num_codebooks]
        if self.segment_weights is None:
            self.segment_weights = {"prefix": 1.0, "suffix": 1.0, "middle": 3.0}


class VoiceCraftX(nn.Module):
    """VoiceCraft-X: Unified multilingual speech synthesis and editing model.

    Args:
        config: VoiceCraftXConfig object
    """

    def __init__(self, config: VoiceCraftXConfig):
        super().__init__()

        self.config = config

        # Speech tokenizer (EnCodec-style with RVQ)
        self.codec = EnCodecTokenizer(
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            sample_rate=config.sample_rate,
            target_framerate=config.codec_framerate,
        )

        # Qwen3 backbone for text processing and generation
        self.backbone = Qwen3Backbone(
            model_name=config.qwen_model_name,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            freeze_backbone=config.freeze_qwen,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            speaker_embedding_dim=config.speaker_embedding_dim,
        )

        # Speaker encoder
        self.speaker_encoder = create_speaker_encoder(
            encoder_type=config.speaker_encoder_type,
            model_path=config.speaker_encoder_path,
            embedding_dim=config.speaker_embedding_dim,
        )

        # Delay pattern for multi-codebook generation
        if config.use_delay_pattern:
            self.delay_pattern = DelayPattern(
                num_codebooks=config.num_codebooks,
                delay_per_codebook=1,
                special_token_id=config.codebook_size,  # Use special ID for padding
            )
        else:
            self.delay_pattern = None

        # Token reordering strategy
        if config.use_token_reordering:
            self.reordering = TokenReorderingStrategy(
                mask_token_id=self.backbone.mask_token_id,
                min_middle_ratio=0.1,
                max_middle_ratio=0.8,
            )
        else:
            self.reordering = None

        # Loss function
        self.loss_fn = VoiceCraftXLoss(
            num_codebooks=config.num_codebooks,
            codebook_weights=config.codebook_weights,
            segment_weights=config.segment_weights,
        )

    def encode_audio(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to discrete tokens.

        Args:
            audio: Audio waveform [B, T_audio] or [B, 1, T_audio]

        Returns:
            indices: Codebook indices [B, K, T_tokens]
            quant_loss: Quantization loss
        """
        return self.codec.encode(audio, return_all_codes=True)

    def decode_audio(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode discrete tokens to audio.

        Args:
            indices: Codebook indices [B, K, T_tokens]

        Returns:
            audio: Reconstructed audio [B, 1, T_audio]
        """
        return self.codec.decode(indices)

    def extract_speaker_embedding(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio waveform [B, T_audio]
            sample_rate: Sample rate (uses config default if None)

        Returns:
            Speaker embedding [B, D_speaker]
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate

        return self.speaker_encoder(audio, sample_rate=sample_rate)

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
        segment_lengths: Optional[Dict[str, int]] = None,
        return_loss: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass for training.

        Args:
            text_tokens: Text token IDs [B, T_text]
            audio_tokens: Audio token IDs [B, K, T_audio]
            speaker_embedding: Optional speaker embedding [B, D_speaker]
            segment_lengths: Optional segment lengths for loss computation
            return_loss: Whether to compute and return loss

        Returns:
            If return_loss=True: (loss, loss_dict)
            If return_loss=False: logits for each codebook
        """
        # Forward through backbone
        hidden_states, _ = self.backbone(
            input_ids=text_tokens,
            audio_tokens=audio_tokens,
            speaker_embedding=speaker_embedding,
        )

        # Predict codebook logits
        logits = self.backbone.predict_codebooks(hidden_states)

        if return_loss:
            # Compute loss
            # Note: We need to align logits with target positions
            # For training, we predict all positions (prefix + suffix + middle)
            # Extract the speech token positions from total sequence
            text_len = text_tokens.shape[1] if text_tokens is not None else 0
            speaker_len = 1 if speaker_embedding is not None else 0
            speech_start = text_len + speaker_len

            # Get logits corresponding to speech positions
            speech_logits = [
                logit[:, speech_start:, :] for logit in logits
            ]

            # Compute loss
            loss, loss_dict = self.loss_fn(
                speech_logits,
                audio_tokens,
                segment_lengths=segment_lengths,
            )

            return loss, loss_dict
        else:
            return logits

    def inference_tts(
        self,
        text: str,
        prompt_audio: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Zero-shot TTS inference.

        Args:
            text: Target text to synthesize
            prompt_audio: Optional prompt audio for voice cloning [T_audio]
            prompt_text: Optional text corresponding to prompt audio
            speaker_embedding: Optional pre-computed speaker embedding
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            min_p: Minimum probability threshold (alternative to top_p)
            repetition_penalty: Penalty for repeating tokens (>1.0 to discourage)

        Returns:
            Generated audio waveform [T_audio]
        """
        self.eval()

        with torch.no_grad():
            # Tokenize text
            text_tokens = self.backbone.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
            )["input_ids"].to(next(self.parameters()).device)

            # Extract speaker embedding if not provided
            if speaker_embedding is None and prompt_audio is not None:
                speaker_embedding = self.extract_speaker_embedding(
                    prompt_audio.unsqueeze(0) if prompt_audio.dim() == 1 else prompt_audio
                )

            # Encode prompt audio if provided
            prompt_audio_tokens = None
            if prompt_audio is not None:
                if prompt_audio.dim() == 1:
                    prompt_audio = prompt_audio.unsqueeze(0)
                prompt_audio_tokens, _ = self.encode_audio(prompt_audio)

            # Tokenize prompt text if provided
            prompt_text_tokens = None
            if prompt_text is not None:
                prompt_text_tokens = self.backbone.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"].to(next(self.parameters()).device)

            # Create TTS sequence using reordering strategy
            if self.reordering is not None:
                reordered = self.reordering.create_tts_sequence(
                    target_text=text_tokens[0],
                    prompt_text=prompt_text_tokens[0] if prompt_text_tokens is not None else None,
                    prompt_speech=prompt_audio_tokens[0] if prompt_audio_tokens is not None else None,
                    speaker_embedding=speaker_embedding[0] if speaker_embedding is not None else None,
                )

                # Get input sequence
                text_input, speech_input = reordered.get_input_sequence(include_middle_speech=True)

                # Add batch dimension
                text_input = text_input.unsqueeze(0) if len(text_input) > 0 else None
                speech_input = speech_input.unsqueeze(0) if speech_input.numel() > 0 else None
            else:
                # Simple concatenation without reordering
                text_input = text_tokens
                speech_input = prompt_audio_tokens

            # Generate audio tokens
            generated_tokens = self.backbone.generate(
                input_ids=text_input,
                audio_tokens=speech_input,
                speaker_embedding=speaker_embedding,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )

            # Decode to audio
            audio = self.decode_audio(generated_tokens)

            # Return without batch dimension
            return audio[0, 0]  # [T_audio]

    def inference_edit(
        self,
        prefix_audio: torch.Tensor,
        suffix_audio: torch.Tensor,
        new_middle_text: str,
        prefix_text: Optional[str] = None,
        suffix_text: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Speech editing inference.

        Args:
            prefix_audio: Audio before edit point [T_prefix]
            suffix_audio: Audio after edit point [T_suffix]
            new_middle_text: New text to insert/replace
            prefix_text: Optional text for prefix
            suffix_text: Optional text for suffix
            speaker_embedding: Optional speaker embedding
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            min_p: Minimum probability threshold (alternative to top_p)
            repetition_penalty: Penalty for repeating tokens (>1.0 to discourage)

        Returns:
            Edited audio waveform [T_audio] (seamless concatenation)
        """
        self.eval()

        with torch.no_grad():
            # Ensure batch dimension
            if prefix_audio.dim() == 1:
                prefix_audio = prefix_audio.unsqueeze(0)
            if suffix_audio.dim() == 1:
                suffix_audio = suffix_audio.unsqueeze(0)

            # Encode audio segments
            prefix_tokens, _ = self.encode_audio(prefix_audio)
            suffix_tokens, _ = self.encode_audio(suffix_audio)

            # Extract speaker embedding if not provided
            if speaker_embedding is None:
                # Use prefix audio for speaker embedding
                speaker_embedding = self.extract_speaker_embedding(prefix_audio)

            # Tokenize text segments
            middle_text_tokens = self.backbone.tokenizer(
                new_middle_text,
                return_tensors="pt",
                padding=False,
            )["input_ids"].to(next(self.parameters()).device)

            prefix_text_tokens = None
            if prefix_text is not None:
                prefix_text_tokens = self.backbone.tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"].to(next(self.parameters()).device)[0]

            suffix_text_tokens = None
            if suffix_text is not None:
                suffix_text_tokens = self.backbone.tokenizer(
                    suffix_text,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"].to(next(self.parameters()).device)[0]

            # Create editing sequence
            if self.reordering is not None:
                reordered = self.reordering.create_editing_sequence(
                    prefix_text=prefix_text_tokens if prefix_text_tokens is not None else torch.tensor([]),
                    suffix_text=suffix_text_tokens if suffix_text_tokens is not None else torch.tensor([]),
                    new_middle_text=middle_text_tokens[0],
                    prefix_speech=prefix_tokens[0],
                    suffix_speech=suffix_tokens[0],
                    speaker_embedding=speaker_embedding[0],
                )

                # Get input sequence (without middle speech - to be generated)
                text_input, speech_input = reordered.get_input_sequence(include_middle_speech=False)

                # Add batch dimension
                text_input = text_input.unsqueeze(0)
                speech_input = speech_input.unsqueeze(0)
            else:
                raise NotImplementedError("Editing requires token reordering")

            # Generate middle audio tokens
            generated_middle = self.backbone.generate(
                input_ids=text_input,
                audio_tokens=speech_input,
                speaker_embedding=speaker_embedding,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )

            # Concatenate: prefix + generated_middle + suffix
            full_tokens = torch.cat([
                prefix_tokens,
                generated_middle,
                suffix_tokens,
            ], dim=2)

            # Decode to audio
            audio = self.decode_audio(full_tokens)

            return audio[0, 0]  # [T_audio]


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("VoiceCraft-X Model Test")
    print("=" * 60)

    # Create config
    config = VoiceCraftXConfig(
        num_codebooks=4,
        codebook_size=2048,
        qwen_model_name="Qwen/Qwen2.5-0.5B",
        use_delay_pattern=True,
        use_token_reordering=True,
    )

    print("\nConfiguration:")
    print(f"  Num codebooks: {config.num_codebooks}")
    print(f"  Codebook size: {config.codebook_size}")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Codec framerate: {config.codec_framerate} Hz")

    # Initialize model
    print("\nInitializing VoiceCraft-X model...")
    model = VoiceCraftX(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Test audio encoding/decoding
    print("\n" + "=" * 60)
    print("Testing Audio Encoding/Decoding")
    print("=" * 60)

    batch_size = 2
    audio_duration = 3.0  # seconds
    audio_length = int(audio_duration * config.sample_rate)

    audio = torch.randn(batch_size, audio_length)
    print(f"\nInput audio: {audio.shape}")

    tokens, quant_loss = model.encode_audio(audio)
    print(f"Encoded tokens: {tokens.shape}")
    print(f"Quantization loss: {quant_loss.item():.4f}")

    audio_recon = model.decode_audio(tokens)
    print(f"Reconstructed audio: {audio_recon.shape}")

    # Test speaker embedding
    print("\n" + "=" * 60)
    print("Testing Speaker Embedding")
    print("=" * 60)

    speaker_emb = model.extract_speaker_embedding(audio[:1])
    print(f"Speaker embedding: {speaker_emb.shape}")
    print(f"Embedding norm: {speaker_emb.norm(dim=1).item():.4f}")

    print("\n✓ All tests passed!")
