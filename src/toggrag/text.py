import re

_TR_MAKE_TRANS = str.maketrans("İI", "iı")
_WORD = re.compile(r"\w+")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I -> ı and İ -> i (str.lower() gets both wrong)."""
    return text.translate(_TR_MAKE_TRANS).lower()


def tokenize(text: str) -> list[str]:
    """Split into lowercase Turkish words; digits kept, punctuation dropped."""
    return [turkish_lower(word) for word in _WORD.findall(text)]
