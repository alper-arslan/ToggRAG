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


def clean_page_text(text: str, page: int) -> str:
    """Remove picture noise that PDF extraction mixes into the page text.

    The text layer includes lines that come from the figures, not from the
    manual's prose. Left in, they would end up inside chunks and pollute the
    search index. Removed:
    - figure id lines (TG02307, TG02816_T),
    - digit-only lines right after a figure id (the numbers drawn on the picture),
    - the printed page number on the last line.

    Digit-only lines elsewhere are kept: they can be real values, such as the
    vehicle dimensions on the specification pages.
    """
    kept = []
    in_labels = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TG") and line.removesuffix("_T")[2:].isdigit():
            in_labels = True
        elif in_labels and line.replace(" ", "").isdigit():
            pass
        else:
            in_labels = False
            kept.append(line)
    if kept and kept[-1] == str(page):
        kept.pop()
    return "\n".join(kept)
