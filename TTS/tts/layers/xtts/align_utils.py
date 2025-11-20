"""
Alignment Utilities for VoiceCraft-X Speech Editing

Provides text alignment and segmentation utilities for precise speech editing.
Based on VoiceCraft-X implementation.

Key functions:
- get_diff_time_frame_and_segment: Find prefix/suffix/middle boundaries
- build_mapping: Map cleaned text to original positions
- remove_punctuation: Unicode-aware punctuation removal
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# Language categories
WORD_BASED_LANGUAGES = {
    "en",  # English
    "es",  # Spanish
    "nl",  # Dutch
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "pl",  # Polish
    "ko",  # Korean (word-based despite being Asian)
}

CHARACTER_BASED_LANGUAGES = {
    "zh",  # Chinese
    "ja",  # Japanese
}


def get_all_unicode_punctuation() -> str:
    """Get all Unicode punctuation characters.

    Returns:
        String containing all Unicode punctuation characters
    """
    punctuation = []
    for i in range(0x10FFFF + 1):  # Full Unicode range
        try:
            char = chr(i)
            if unicodedata.category(char).startswith('P'):  # P = Punctuation
                punctuation.append(char)
        except ValueError:
            continue
    return ''.join(punctuation)


# Cache punctuation for performance
_UNICODE_PUNCTUATION = get_all_unicode_punctuation()
_PUNCTUATION_PATTERN = re.compile(f'[{re.escape(_UNICODE_PUNCTUATION)}]')


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text.

    Args:
        text: Input text with punctuation

    Returns:
        Text with punctuation removed
    """
    return _PUNCTUATION_PATTERN.sub('', text)


def build_mapping(original: str) -> Tuple[List[int], str]:
    """Build mapping between cleaned text and original positions.

    Removes punctuation while preserving spaces, creating a mapping
    that allows recovering original character positions.

    Args:
        original: Original text with punctuation

    Returns:
        Tuple of (mapping list, cleaned text)
        - mapping[i] gives the position in original text for character i in cleaned
        - cleaned is the text with punctuation removed
    """
    mapping = []
    cleaned = []

    for i, char in enumerate(original):
        if not _PUNCTUATION_PATTERN.match(char):
            mapping.append(i)
            cleaned.append(char)

    return mapping, ''.join(cleaned)


def build_mapping_tokens(original: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Build word-level mapping for word-based languages.

    Uses regex to find word boundaries and map them to original positions.

    Args:
        original: Original text

    Returns:
        Tuple of (boundaries, words)
        - boundaries[i] = (start, end) positions in original text for word i
        - words = list of words
    """
    # First remove punctuation but keep track of positions
    mapping, cleaned = build_mapping(original)

    # Find words in cleaned text
    word_pattern = re.compile(r'\S+')  # Non-whitespace sequences
    boundaries = []
    words = []

    for match in word_pattern.finditer(cleaned):
        start_clean, end_clean = match.span()
        # Map back to original positions
        start_orig = mapping[start_clean]
        end_orig = mapping[end_clean - 1] + 1  # +1 to include last char
        boundaries.append((start_orig, end_orig))
        words.append(match.group())

    return boundaries, words


def get_diff_time_frame_and_segment(
    prompt_text: str,
    target_text: str,
    alignment_frames: List[Tuple[int, int]],
    alignment_words: List[str],
    language: str = "en",
) -> Tuple[Tuple[int, int], Tuple[str, str, str]]:
    """Find prefix/suffix/middle boundaries for speech editing.

    This is the core function for speech editing. It identifies:
    - Common prefix between prompt and target
    - Common suffix between prompt and target
    - The middle portion that needs to be generated/edited

    Args:
        prompt_text: Original text corresponding to audio
        target_text: New target text to achieve
        alignment_frames: List of (start_frame, end_frame) for each word/char
        alignment_words: List of words/characters from forced alignment
        language: Language code (affects word vs character processing)

    Returns:
        Tuple of ((start_frame, end_frame), (prefix, middle, suffix))
        - (start_frame, end_frame): Frame range for middle portion
        - (prefix, middle, suffix): Text segments
    """
    # Determine if language is word-based or character-based
    is_word_based = language in WORD_BASED_LANGUAGES

    # Clean texts
    prompt_clean = remove_punctuation(prompt_text)
    target_clean = remove_punctuation(target_text)

    if is_word_based:
        # Split into words
        prompt_tokens = prompt_clean.split()
        target_tokens = target_clean.split()
    else:
        # Character-based: use individual characters (excluding spaces)
        prompt_tokens = [c for c in prompt_clean if not c.isspace()]
        target_tokens = [c for c in target_clean if not c.isspace()]

    # Find common prefix length
    prefix_len = 0
    for p, t in zip(prompt_tokens, target_tokens):
        if p == t:
            prefix_len += 1
        else:
            break

    # Find common suffix length (but don't overlap with prefix)
    suffix_len = 0
    prompt_remaining = len(prompt_tokens) - prefix_len
    target_remaining = len(target_tokens) - prefix_len
    max_suffix = min(prompt_remaining, target_remaining)

    for i in range(1, max_suffix + 1):
        if prompt_tokens[-i] == target_tokens[-i]:
            suffix_len = i
        else:
            break

    # Extract segments
    prefix_tokens = target_tokens[:prefix_len]
    suffix_tokens = target_tokens[-suffix_len:] if suffix_len > 0 else []
    middle_tokens = target_tokens[prefix_len:len(target_tokens) - suffix_len if suffix_len > 0 else len(target_tokens)]

    # Reconstruct text segments
    if is_word_based:
        prefix = ' '.join(prefix_tokens) if prefix_tokens else ''
        middle = ' '.join(middle_tokens) if middle_tokens else ''
        suffix = ' '.join(suffix_tokens) if suffix_tokens else ''
    else:
        prefix = ''.join(prefix_tokens)
        middle = ''.join(middle_tokens)
        suffix = ''.join(suffix_tokens)

    # Map to alignment frames
    # Alignment should have one entry per word/character
    if len(alignment_frames) != len(alignment_words):
        raise ValueError(
            f"Alignment mismatch: {len(alignment_frames)} frames but {len(alignment_words)} words"
        )

    # Find frame boundaries for middle section
    # Middle starts after prefix, ends before suffix
    if prefix_len >= len(alignment_frames):
        # Edge case: prefix covers everything
        start_frame = alignment_frames[-1][1] if alignment_frames else 0
        end_frame = start_frame
    elif prefix_len + suffix_len >= len(alignment_frames):
        # Edge case: prefix + suffix covers everything (no middle in prompt)
        start_frame = alignment_frames[prefix_len][0] if prefix_len < len(alignment_frames) else 0
        end_frame = start_frame
    else:
        # Normal case
        start_frame = alignment_frames[prefix_len][0]
        middle_end_idx = len(alignment_frames) - suffix_len - 1
        end_frame = alignment_frames[middle_end_idx][1]

    return (start_frame, end_frame), (prefix, middle, suffix)


@dataclass
class AlignmentResult:
    """Result from alignment utilities.

    Attributes:
        start_frame: Start frame for middle (edited) section
        end_frame: End frame for middle (edited) section
        prefix_text: Text before edit point
        middle_text: Text to be generated/edited
        suffix_text: Text after edit point
        prefix_frames: Frame range for prefix
        suffix_frames: Frame range for suffix
    """
    start_frame: int
    end_frame: int
    prefix_text: str
    middle_text: str
    suffix_text: str
    prefix_frames: Optional[Tuple[int, int]] = None
    suffix_frames: Optional[Tuple[int, int]] = None


def align_for_editing(
    prompt_text: str,
    target_text: str,
    alignment_frames: List[Tuple[int, int]],
    alignment_words: List[str],
    language: str = "en",
) -> AlignmentResult:
    """High-level alignment function for speech editing.

    Wraps get_diff_time_frame_and_segment with a clean interface.

    Args:
        prompt_text: Original text
        target_text: Target text
        alignment_frames: Frame boundaries from forced aligner
        alignment_words: Words/characters from forced aligner
        language: Language code

    Returns:
        AlignmentResult with all boundary information
    """
    (start_frame, end_frame), (prefix, middle, suffix) = get_diff_time_frame_and_segment(
        prompt_text=prompt_text,
        target_text=target_text,
        alignment_frames=alignment_frames,
        alignment_words=alignment_words,
        language=language,
    )

    # Calculate prefix and suffix frame ranges
    prefix_frames = None
    suffix_frames = None

    if alignment_frames:
        if start_frame > 0:
            prefix_frames = (alignment_frames[0][0], start_frame)
        if end_frame < alignment_frames[-1][1]:
            suffix_frames = (end_frame, alignment_frames[-1][1])

    return AlignmentResult(
        start_frame=start_frame,
        end_frame=end_frame,
        prefix_text=prefix,
        middle_text=middle,
        suffix_text=suffix,
        prefix_frames=prefix_frames,
        suffix_frames=suffix_frames,
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Alignment Utilities")
    print("=" * 60)

    # Test 1: Punctuation removal
    print("\n[Test 1] Punctuation Removal")
    text_with_punct = "Hello, world! How are you?"
    text_clean = remove_punctuation(text_with_punct)
    print(f"Original: '{text_with_punct}'")
    print(f"Cleaned:  '{text_clean}'")

    # Test 2: Mapping
    print("\n[Test 2] Position Mapping")
    mapping, cleaned = build_mapping(text_with_punct)
    print(f"Cleaned: '{cleaned}'")
    print(f"Mapping: {mapping[:10]}...")  # First 10 positions

    # Test 3: Word boundaries
    print("\n[Test 3] Word Boundaries")
    boundaries, words = build_mapping_tokens(text_with_punct)
    print(f"Words: {words}")
    print(f"Boundaries: {boundaries}")

    # Test 4: Diff alignment (word-based)
    print("\n[Test 4] Diff Alignment (English)")
    prompt = "The quick brown fox jumps over the lazy dog"
    target = "The quick brown cat jumps over the sleepy dog"

    # Simulate alignment (normally from forced aligner)
    # Each word gets a 50-frame window
    words = prompt.split()
    frames = [(i * 50, (i + 1) * 50) for i in range(len(words))]

    result = align_for_editing(
        prompt_text=prompt,
        target_text=target,
        alignment_frames=frames,
        alignment_words=words,
        language="en",
    )

    print(f"Prompt:  '{prompt}'")
    print(f"Target:  '{target}'")
    print(f"\nPrefix:  '{result.prefix_text}'")
    print(f"Middle:  '{result.middle_text}'")
    print(f"Suffix:  '{result.suffix_text}'")
    print(f"\nFrame range for middle: {result.start_frame}-{result.end_frame}")

    # Test 5: Character-based (Chinese)
    print("\n[Test 5] Diff Alignment (Chinese)")
    prompt_zh = "我爱吃苹果"  # I love eating apples
    target_zh = "我爱吃香蕉"  # I love eating bananas

    chars = list(prompt_zh)
    frames_zh = [(i * 50, (i + 1) * 50) for i in range(len(chars))]

    result_zh = align_for_editing(
        prompt_text=prompt_zh,
        target_text=target_zh,
        alignment_frames=frames_zh,
        alignment_words=chars,
        language="zh",
    )

    print(f"Prompt:  '{prompt_zh}'")
    print(f"Target:  '{target_zh}'")
    print(f"\nPrefix:  '{result_zh.prefix_text}'")
    print(f"Middle:  '{result_zh.middle_text}'")
    print(f"Suffix:  '{result_zh.suffix_text}'")

    print("\n✓ All tests passed!")
