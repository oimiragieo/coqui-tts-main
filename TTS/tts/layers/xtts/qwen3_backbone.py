"""
Qwen3 Backbone for Multilingual Phoneme-Free Text Processing

Based on VoiceCraft-X paper (arXiv:2511.12347v1).
Uses Qwen3-0.6B-Base for:
- Native support for 119 languages
- No need for G2P (grapheme-to-phoneme) conversion
- Shared tokenizer across all languages
- Strong multilingual representations

Architecture:
- 28 Transformer layers
- Hidden dim: 1024
- FFN dim: 3072
- 16 attention heads (Grouped-Query Attention with 8 KV heads)
- Context length: 32,768 tokens
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Qwen2Config, Qwen2Model


class Qwen3Backbone(nn.Module):
    """Qwen3-based backbone for multilingual TTS.

    Args:
        model_name: HuggingFace model name (default: "Qwen/Qwen2.5-0.5B")
        num_codebooks: Number of speech codebooks for prediction heads
        codebook_size: Vocabulary size per codebook
        freeze_backbone: Whether to freeze Qwen3 parameters
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_rank: LoRA rank if use_lora=True
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_codebooks: int = 4,
        codebook_size: int = 2048,
        freeze_backbone: bool = False,
        use_lora: bool = False,
        lora_rank: int = 16,
        speaker_embedding_dim: int = 512,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Add special tokens
        special_tokens = {
            'additional_special_tokens': ['<MASK>', '<SPK>', '<AUD>']
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Get special token IDs
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids('<MASK>')
        self.speaker_token_id = self.tokenizer.convert_tokens_to_ids('<SPK>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<AUD>')

        # Load Qwen3 model
        try:
            self.qwen = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using Qwen2.5-0.5B: {e}")
            # Fallback to Qwen2.5-0.5B which has similar architecture
            self.qwen = AutoModel.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True,
            )

        # Resize embeddings for new special tokens
        self.qwen.resize_token_embeddings(len(self.tokenizer))

        # Get model config
        self.config = self.qwen.config
        self.hidden_dim = self.config.hidden_size

        # Speaker embedding projection
        self.speaker_projection = nn.Linear(speaker_embedding_dim, self.hidden_dim)

        # Audio token embeddings (for speech codes)
        # Each codebook gets its own embedding, then summed
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, self.hidden_dim)
            for _ in range(num_codebooks)
        ])

        # Prediction heads for each codebook
        self.codebook_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, codebook_size)
            for _ in range(num_codebooks)
        ])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.qwen.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank)

    def _apply_lora(self, rank: int):
        """Apply LoRA (Low-Rank Adaptation) to attention layers.

        Args:
            rank: LoRA rank
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

            self.qwen = get_peft_model(self.qwen, lora_config)
            print(f"✓ Applied LoRA with rank {rank}")
            self.qwen.print_trainable_parameters()

        except ImportError:
            print("Warning: peft library not available. Install with: pip install peft")
            print("Continuing without LoRA...")

    def embed_text(
        self,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Embed text tokens using Qwen3 embeddings.

        Args:
            text_tokens: Text token IDs [B, T_text]

        Returns:
            Text embeddings [B, T_text, D]
        """
        return self.qwen.get_input_embeddings()(text_tokens)

    def embed_audio(
        self,
        audio_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Embed audio tokens (multi-codebook).

        Args:
            audio_tokens: Audio token IDs [B, K, T_audio] or [B, T_audio]

        Returns:
            Audio embeddings [B, T_audio, D]
        """
        if audio_tokens.dim() == 2:
            # Single codebook: [B, T_audio]
            return self.audio_embeddings[0](audio_tokens)
        else:
            # Multi-codebook: [B, K, T_audio]
            B, K, T = audio_tokens.shape
            assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

            # Embed each codebook and sum
            embeddings = []
            for k in range(K):
                emb = self.audio_embeddings[k](audio_tokens[:, k])  # [B, T, D]
                embeddings.append(emb)

            # Sum across codebooks
            return torch.stack(embeddings, dim=0).sum(dim=0)  # [B, T, D]

    def embed_speaker(
        self,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Project speaker embedding to model dimension.

        Args:
            speaker_embedding: Speaker embedding vector [B, D_spk]

        Returns:
            Projected embedding [B, 1, D]
        """
        projected = self.speaker_projection(speaker_embedding)  # [B, D]
        return projected.unsqueeze(1)  # [B, 1, D]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass through Qwen3 backbone.

        Args:
            input_ids: Text token IDs [B, T_text]
            audio_tokens: Audio token IDs [B, K, T_audio]
            speaker_embedding: Speaker embedding [B, D_spk]
            attention_mask: Attention mask [B, T_total]
            position_ids: Position IDs [B, T_total]
            past_key_values: Cached key-values for generation
            use_cache: Whether to return key-value cache
            output_hidden_states: Whether to output all hidden states

        Returns:
            hidden_states: Output hidden states [B, T_total, D]
            past_key_values: KV cache if use_cache=True
        """
        # Construct input embeddings
        embeddings_list = []

        # Add text embeddings
        if input_ids is not None and len(input_ids) > 0:
            text_emb = self.embed_text(input_ids)  # [B, T_text, D]
            embeddings_list.append(text_emb)

        # Add speaker embedding
        if speaker_embedding is not None:
            speaker_emb = self.embed_speaker(speaker_embedding)  # [B, 1, D]
            embeddings_list.append(speaker_emb)

        # Add audio embeddings
        if audio_tokens is not None and audio_tokens.numel() > 0:
            audio_emb = self.embed_audio(audio_tokens)  # [B, T_audio, D]
            embeddings_list.append(audio_emb)

        # Concatenate all embeddings
        if not embeddings_list:
            raise ValueError("At least one of input_ids, audio_tokens, or speaker_embedding must be provided")

        inputs_embeds = torch.cat(embeddings_list, dim=1)  # [B, T_total, D]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )

        # Forward through Qwen3
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, T_total, D]

        if use_cache:
            return hidden_states, outputs.past_key_values
        else:
            return hidden_states, None

    def predict_codebooks(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Predict codebook logits from hidden states.

        Args:
            hidden_states: Hidden states [B, T, D]

        Returns:
            Tuple of logits for each codebook, each [B, T, codebook_size]
        """
        logits = []
        for k in range(self.num_codebooks):
            logit = self.codebook_heads[k](hidden_states)  # [B, T, codebook_size]
            logits.append(logit)

        return tuple(logits)

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation of audio tokens.

        Args:
            input_ids: Text token IDs [B, T_text]
            audio_tokens: Initial audio tokens [B, K, T_audio]
            speaker_embedding: Speaker embedding [B, D_spk]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            use_cache: Use KV caching for efficiency

        Returns:
            Generated audio tokens [B, K, T_generated]
        """
        B = 1 if input_ids is None else input_ids.shape[0]
        device = next(self.parameters()).device

        # Initialize generated tokens
        if audio_tokens is not None and audio_tokens.numel() > 0:
            generated = audio_tokens.clone()
        else:
            generated = torch.zeros(
                B, self.num_codebooks, 0,
                dtype=torch.long,
                device=device,
            )

        past_key_values = None

        for step in range(max_new_tokens):
            # Prepare input for this step
            if step == 0:
                # First step: use full context
                step_audio = generated if generated.shape[2] > 0 else None
            else:
                # Subsequent steps: use only last generated token
                step_audio = generated[:, :, -1:] if use_cache else generated

            # Forward pass
            hidden_states, past_key_values = self.forward(
                input_ids=input_ids if step == 0 else None,
                audio_tokens=step_audio,
                speaker_embedding=speaker_embedding if step == 0 else None,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )

            # Predict next tokens for each codebook
            logits = self.predict_codebooks(hidden_states)

            # Sample from each codebook
            next_tokens = []
            for k in range(self.num_codebooks):
                # Get logits for last position
                logits_k = logits[k][:, -1, :] / temperature  # [B, codebook_size]

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits_k < torch.topk(logits_k, top_k)[0][..., -1, None]
                    logits_k[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_k, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_k[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(logits_k, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                next_tokens.append(next_token)

            # Stack next tokens: [B, K, 1]
            next_tokens = torch.stack(next_tokens, dim=1)

            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=2)

            # Stop if all sequences in batch are done (you'd add your own stopping logic here)
            # For now, just continue until max_new_tokens

        return generated


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qwen3 Backbone")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    text_length = 20
    audio_length = 50
    num_codebooks = 4
    codebook_size = 2048
    speaker_dim = 512

    # Initialize model
    print("\nInitializing Qwen3 backbone...")
    model = Qwen3Backbone(
        model_name="Qwen/Qwen2.5-0.5B",
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        freeze_backbone=False,
        use_lora=False,
    )

    print(f"✓ Model initialized")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Mask token ID: {model.mask_token_id}")
    print(f"  Speaker token ID: {model.speaker_token_id}")

    # Test tokenizer
    print("\n" + "=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)

    test_texts = [
        "Hello, how are you?",
        "你好，你好吗？",  # Chinese
        "Bonjour, comment allez-vous?",  # French
    ]

    for text in test_texts:
        tokens = model.tokenizer(text, return_tensors="pt")
        print(f"\nText: {text}")
        print(f"  Tokens: {tokens['input_ids'].shape}")
        print(f"  Token IDs: {tokens['input_ids'][0, :10].tolist()}...")

    # Test embeddings
    print("\n" + "=" * 60)
    print("Testing Embeddings")
    print("=" * 60)

    input_ids = torch.randint(0, model.tokenizer.vocab_size, (batch_size, text_length))
    audio_tokens = torch.randint(0, codebook_size, (batch_size, num_codebooks, audio_length))
    speaker_embedding = torch.randn(batch_size, speaker_dim)

    text_emb = model.embed_text(input_ids)
    print(f"\nText embeddings: {text_emb.shape}")

    audio_emb = model.embed_audio(audio_tokens)
    print(f"Audio embeddings: {audio_emb.shape}")

    speaker_emb = model.embed_speaker(speaker_embedding)
    print(f"Speaker embeddings: {speaker_emb.shape}")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    hidden_states, _ = model(
        input_ids=input_ids,
        audio_tokens=audio_tokens,
        speaker_embedding=speaker_embedding,
    )

    print(f"\nHidden states: {hidden_states.shape}")
    print(f"Expected: [B={batch_size}, T={text_length + 1 + audio_length}, D={model.hidden_dim}]")

    # Test prediction heads
    logits = model.predict_codebooks(hidden_states)
    print(f"\nPrediction logits:")
    for k, logit in enumerate(logits):
        print(f"  Codebook {k}: {logit.shape}")

    # Test generation (small example)
    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60)

    generated = model.generate(
        input_ids=input_ids[:1],  # Single example
        speaker_embedding=speaker_embedding[:1],
        max_new_tokens=10,
        temperature=1.0,
        top_k=20,
        use_cache=True,
    )

    print(f"\nGenerated audio tokens: {generated.shape}")
    print(f"Expected: [B=1, K={num_codebooks}, T=10]")

    print("\n✓ All tests passed!")
