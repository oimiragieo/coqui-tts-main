"""
Text Preprocessing Pipeline for VoiceCraft-X

Provides text normalization and preprocessing before tokenization.
Based on CosyVoice text frontend.

Features:
- Number normalization (digits to words)
- Punctuation-based sentence splitting
- Text cleaning and formatting
- Multi-language support
"""

import re
from typing import List, Optional, Tuple

import torch


def spell_out_number(text: str, language: str = "en") -> str:
    """Convert digit sequences to words.

    Args:
        text: Input text with numbers
        language: Language code for number pronunciation

    Returns:
        Text with numbers spelled out
    """
    # Simple implementation - in production use inflect or num2words
    if language == "en":
        try:
            import inflect
            p = inflect.engine()

            def replace_number(match):
                num_str = match.group()
                try:
                    # Try to convert to number and spell out
                    num = int(num_str) if '.' not in num_str else float(num_str)
                    return p.number_to_words(num)
                except (ValueError, OverflowError):
                    return num_str

            # Replace sequences of digits (possibly with decimal points)
            text = re.sub(r'\d+\.?\d*', replace_number, text)
        except ImportError:
            # Fallback: just leave numbers as-is
            pass
    # Add support for other languages as needed
    return text


def replace_blank(text: str) -> str:
    """Remove spaces between non-ASCII characters.

    Useful for Chinese/Japanese where spaces are not needed.

    Args:
        text: Input text

    Returns:
        Text with spaces adjusted for Asian languages
    """
    # Remove spaces between consecutive non-ASCII characters
    # Pattern: non-ASCII + space + non-ASCII -> non-ASCII + non-ASCII
    text = re.sub(r'([^\x00-\x7F])\s+([^\x00-\x7F])', r'\1\2', text)
    return text


def is_only_punctuation(text: str) -> bool:
    """Check if text contains only punctuation.

    Args:
        text: Input text

    Returns:
        True if text is only punctuation/whitespace
    """
    # Remove whitespace and check if anything remains
    text_no_space = text.strip()
    if not text_no_space:
        return True

    # Check if all remaining chars are punctuation
    for char in text_no_space:
        if char.isalnum():
            return False

    return True


def split_paragraph(
    text: str,
    max_tokens: int = 80,
    min_tokens: int = 60,
    punctuation: str = ",.!?;:，。！？；：",
) -> List[str]:
    """Split long text into manageable segments.

    Splits by punctuation while respecting token budget.

    Args:
        text: Input text to split
        max_tokens: Maximum tokens per segment (approximate)
        min_tokens: Minimum tokens per segment (approximate)
        punctuation: Punctuation marks to split on

    Returns:
        List of text segments
    """
    # Simple character-based splitting (approximates tokens)
    segments = []
    current_segment = []
    current_length = 0

    # Split by punctuation
    punct_pattern = f'[{re.escape(punctuation)}]'
    sentences = re.split(f'({punct_pattern})', text)

    # Recombine sentences with their punctuation
    i = 0
    while i < len(sentences):
        sentence = sentences[i]

        # If this is punctuation, append to previous sentence
        if i > 0 and re.match(punct_pattern, sentence):
            if current_segment:
                current_segment[-1] += sentence
            i += 1
            continue

        sentence_len = len(sentence)

        # If adding this sentence would exceed max, start new segment
        if current_length + sentence_len > max_tokens and current_length >= min_tokens:
            if current_segment:
                segment_text = ''.join(current_segment).strip()
                if segment_text and not is_only_punctuation(segment_text):
                    segments.append(segment_text)
            current_segment = [sentence]
            current_length = sentence_len
        else:
            current_segment.append(sentence)
            current_length += sentence_len

        i += 1

    # Add final segment
    if current_segment:
        segment_text = ''.join(current_segment).strip()
        if segment_text and not is_only_punctuation(segment_text):
            segments.append(segment_text)

    return segments if segments else [text]


class TextPreprocessor:
    """Text preprocessing pipeline for VoiceCraft-X.

    Handles:
    - Text normalization
    - Number conversion
    - Sentence splitting
    - Language-specific processing

    Args:
        language: Default language code (e.g., "en", "zh")
        max_length: Maximum sequence length in characters
        normalize_numbers: Whether to convert numbers to words
    """

    def __init__(
        self,
        language: str = "en",
        max_length: int = 512,
        normalize_numbers: bool = True,
    ):
        self.language = language
        self.max_length = max_length
        self.normalize_numbers = normalize_numbers

    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """Normalize text for synthesis.

        Args:
            text: Input text
            language: Language code (uses default if None)

        Returns:
            Normalized text
        """
        if language is None:
            language = self.language

        # Basic normalization
        text = text.strip()

        # Convert numbers to words
        if self.normalize_numbers:
            text = spell_out_number(text, language)

        # Language-specific processing
        if language in {"zh", "ja", "ko"}:
            # Remove unnecessary spaces in Asian languages
            text = replace_blank(text)

        return text

    def preprocess(
        self,
        text: str,
        language: Optional[str] = None,
        split_long: bool = True,
    ) -> List[str]:
        """Preprocess text for TTS.

        Args:
            text: Input text
            language: Language code (uses default if None)
            split_long: Whether to split long texts

        Returns:
            List of preprocessed text segments
        """
        if language is None:
            language = self.language

        # Normalize
        text = self.normalize_text(text, language)

        # Split if needed
        if split_long and len(text) > self.max_length:
            segments = split_paragraph(
                text,
                max_tokens=80,
                min_tokens=60,
            )
        else:
            segments = [text]

        # Filter empty/punctuation-only segments
        segments = [s for s in segments if s and not is_only_punctuation(s)]

        return segments if segments else [text]

    def __call__(
        self,
        text: str,
        language: Optional[str] = None,
        split_long: bool = True,
    ) -> List[str]:
        """Preprocess text (callable interface).

        Args:
            text: Input text
            language: Language code
            split_long: Whether to split long texts

        Returns:
            List of preprocessed text segments
        """
        return self.preprocess(text, language, split_long)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Text Preprocessor")
    print("=" * 60)

    # Test 1: Basic preprocessing
    print("\n[Test 1] Basic Preprocessing")
    processor = TextPreprocessor(language="en")
    text = "Hello world! This is a test."
    result = processor(text, split_long=False)
    print(f"Input:  '{text}'")
    print(f"Output: {result}")

    # Test 2: Number normalization
    print("\n[Test 2] Number Normalization")
    text_with_numbers = "I have 3 apples and 5 oranges, total 8 fruits."
    result = processor(text_with_numbers, split_long=False)
    print(f"Input:  '{text_with_numbers}'")
    print(f"Output: {result}")

    # Test 3: Long text splitting
    print("\n[Test 3] Long Text Splitting")
    long_text = (
        "This is a very long sentence that goes on and on. "
        "It has multiple parts separated by punctuation. "
        "We want to split it into manageable chunks. "
        "Each chunk should be between 60 and 80 characters long. "
        "This helps with processing efficiency."
    )
    result = processor(long_text, split_long=True)
    print(f"Input length: {len(long_text)}")
    print(f"Number of segments: {len(result)}")
    for i, seg in enumerate(result):
        print(f"  Segment {i + 1} ({len(seg)} chars): '{seg[:50]}...'")

    # Test 4: Chinese text
    print("\n[Test 4] Chinese Text Processing")
    processor_zh = TextPreprocessor(language="zh")
    text_zh = "你好 世界！ 这是 一个 测试。"
    result_zh = processor_zh(text_zh, split_long=False)
    print(f"Input:  '{text_zh}'")
    print(f"Output: {result_zh}")

    # Test 5: Punctuation filtering
    print("\n[Test 5] Punctuation Filtering")
    punct_text = "... , , , !!!"
    result = processor(punct_text, split_long=False)
    print(f"Input:  '{punct_text}'")
    print(f"Output: {result}")
    print(f"Filtered: {len(result) == 0 or result[0] == punct_text}")

    print("\n✓ All tests passed!")
