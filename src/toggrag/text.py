import re

import snowballstemmer

_TR_MAKE_TRANS = str.maketrans("İI", "iı")
_WORD = re.compile(r"\w+")
_STEM = snowballstemmer.stemmer("turkish")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I -> ı and İ -> i (str.lower() gets both wrong)."""
    return text.translate(_TR_MAKE_TRANS).lower()


def tokenize(text: str) -> list[str]:
    """Split into lowercase Turkish words; digits kept, punctuation dropped."""
    return [turkish_lower(word) for word in _WORD.findall(text)]


def stem(word: str) -> str:
    """Reduce a Turkish word to its stem.

    Snowball handles noun suffixes well (ayarlar -> ayar) but leaves many verb
    forms unchanged (ayarlanır stays ayarlanır).
    Input must be lowercase: uppercase words are returned unstemmed.
    analyze() guarantees this.
    """
    return _STEM.stemWord(word)


def analyze(text: str) -> list[str]:
    """Tokenize, lowercase and stem text. Use for both indexing and queries."""
    return [stem(word) for word in tokenize(text)]
