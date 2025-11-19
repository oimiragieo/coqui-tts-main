"""Utilities for PyTorch 2.0+ torch.compile() optimization.

This module provides wrapper functions and decorators to enable torch.compile()
optimization throughout the TTS codebase, with graceful fallback for older PyTorch versions.
"""

import functools
import logging
from typing import Any, Callable, Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Check if torch.compile is available (PyTorch 2.0+)
HAS_TORCH_COMPILE = hasattr(torch, "compile") and callable(getattr(torch, "compile"))


def maybe_compile(
    model: nn.Module,
    mode: str = "default",
    dynamic: Optional[bool] = None,
    fullgraph: bool = False,
    backend: str = "inductor",
    disable: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Conditionally apply torch.compile() to a model.

    This function wraps torch.compile() and gracefully handles cases where:
    - PyTorch version < 2.0 (torch.compile not available)
    - User explicitly disables compilation
    - Compilation fails for any reason

    Args:
        model: PyTorch model to potentially compile
        mode: Compilation mode - "default", "reduce-overhead", "max-autotune"
        dynamic: Enable dynamic shape tracing
        fullgraph: Require full graph capture (stricter but potentially faster)
        backend: Compiler backend to use
        disable: Explicitly disable compilation
        **kwargs: Additional arguments passed to torch.compile()

    Returns:
        Compiled model if successful, otherwise original model

    Example:
        >>> model = MyTTSModel()
        >>> model = maybe_compile(model, mode="reduce-overhead")
        >>> # Model will be compiled if PyTorch >= 2.0, otherwise unchanged
    """
    if disable or not HAS_TORCH_COMPILE:
        if not HAS_TORCH_COMPILE:
            logger.debug("torch.compile() not available (requires PyTorch >= 2.0)")
        return model

    try:
        logger.info(f"Compiling model with mode='{mode}', backend='{backend}'")
        compiled_model = torch.compile(
            model, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend, **kwargs
        )
        logger.info("✓ Model compiled successfully")
        return compiled_model
    except Exception as e:
        logger.warning(f"Failed to compile model: {e}. Using uncompiled model.")
        return model


def compilable(
    mode: str = "default",
    dynamic: Optional[bool] = None,
    fullgraph: bool = False,
    backend: str = "inductor",
    **compile_kwargs: Any,
) -> Callable:
    """Decorator to make a model class compilable.

    This decorator wraps the model's forward method with torch.compile()
    after initialization.

    Args:
        mode: Compilation mode
        dynamic: Enable dynamic shape tracing
        fullgraph: Require full graph capture
        backend: Compiler backend to use
        **compile_kwargs: Additional torch.compile() arguments

    Example:
        >>> @compilable(mode="reduce-overhead")
        >>> class MyTTSModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         # ... model definition
        >>>
        >>>     def forward(self, x):
        >>>         # ... forward pass
        >>>         return output
    """

    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)

            # Apply torch.compile if available
            if HAS_TORCH_COMPILE and not kwargs.get("disable_compile", False):
                try:
                    self.forward = torch.compile(
                        self.forward, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend, **compile_kwargs
                    )
                    logger.info(f"✓ {cls.__name__}.forward() compiled with mode='{mode}'")
                except Exception as e:
                    logger.warning(f"Could not compile {cls.__name__}.forward(): {e}")

        cls.__init__ = new_init
        return cls

    return decorator


def use_fused_attention() -> bool:
    """Check if PyTorch fused scaled_dot_product_attention is available.

    Returns:
        True if torch.nn.functional.scaled_dot_product_attention is available
    """
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Wrapper for scaled dot-product attention with fallback.

    Uses PyTorch's fused implementation if available (faster), otherwise
    falls back to manual implementation.

    Args:
        query: Query tensor of shape (B, ..., L, E)
        key: Key tensor of shape (B, ..., S, E)
        value: Value tensor of shape (B, ..., S, Ev)
        attn_mask: Attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Scaling factor (default: 1/sqrt(E))

    Returns:
        Attention output tensor
    """
    if use_fused_attention():
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    else:
        # Fallback implementation
        embed_dim = query.size(-1)
        if scale is None:
            scale = 1.0 / (embed_dim**0.5)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply causal mask if needed
        if is_causal:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).triu(1)
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Compute attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)

        # Compute output
        output = torch.matmul(attn_probs, value)

        return output


class CompilationConfig:
    """Configuration for torch.compile() settings across the codebase."""

    # Default compilation settings for different model types
    INFERENCE_MODE = "reduce-overhead"  # Optimized for inference latency
    TRAINING_MODE = "default"  # Balanced for training
    AGGRESSIVE_MODE = "max-autotune"  # Maximum optimization (slower compilation)

    # Model-specific settings
    VITS_CONFIG = {"mode": "reduce-overhead", "dynamic": False, "fullgraph": False}

    XTTS_CONFIG = {"mode": "reduce-overhead", "dynamic": True, "fullgraph": False}

    TACOTRON2_CONFIG = {"mode": "default", "dynamic": False, "fullgraph": False}

    VOCODER_CONFIG = {"mode": "reduce-overhead", "dynamic": False, "fullgraph": True}

    @classmethod
    def get_config(cls, model_type: str) -> dict:
        """Get compilation config for a specific model type.

        Args:
            model_type: One of "vits", "xtts", "tacotron2", "vocoder"

        Returns:
            Dictionary of torch.compile() arguments
        """
        configs = {
            "vits": cls.VITS_CONFIG,
            "xtts": cls.XTTS_CONFIG,
            "tacotron2": cls.TACOTRON2_CONFIG,
            "vocoder": cls.VOCODER_CONFIG,
        }
        return configs.get(model_type.lower(), {"mode": "default"})


# Convenience functions for common use cases
def compile_for_inference(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Compile model with settings optimized for inference.

    Args:
        model: Model to compile
        **kwargs: Override default compilation settings

    Returns:
        Compiled model
    """
    config = {"mode": CompilationConfig.INFERENCE_MODE, **kwargs}
    return maybe_compile(model, **config)


def compile_for_training(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Compile model with settings optimized for training.

    Args:
        model: Model to compile
        **kwargs: Override default compilation settings

    Returns:
        Compiled model
    """
    config = {"mode": CompilationConfig.TRAINING_MODE, **kwargs}
    return maybe_compile(model, **config)


# Export public API
__all__ = [
    "HAS_TORCH_COMPILE",
    "maybe_compile",
    "compilable",
    "use_fused_attention",
    "scaled_dot_product_attention",
    "CompilationConfig",
    "compile_for_inference",
    "compile_for_training",
]
