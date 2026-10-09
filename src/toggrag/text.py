_TR_MAKE_TRANS = str.maketrans("İI", "iı")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I -> ı and İ -> i (str.lower() gets both wrong)."""
    return text.translate(_TR_MAKE_TRANS).lower()
