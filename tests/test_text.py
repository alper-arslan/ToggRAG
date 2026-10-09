from toggrag.text import stem, tokenize, turkish_lower


def test_dotted_and_dotless():
    assert turkish_lower("IŞIK") == "ışık"
    assert turkish_lower("İstanbul") == "istanbul"


def test_lower_and_empty_strings_dont_change():
    assert turkish_lower("ışık") == "ışık"
    assert turkish_lower("") == ""


def test_tokenize_split_lowercase_and_drops_punctuation():
    assert tokenize("Aracın IŞIK ayarı nasıl yapılır?") == [
        "aracın",
        "ışık",
        "ayarı",
        "nasıl",
        "yapılır",
    ]


def test_tokenize_keeps_numbers():
    assert tokenize("Lastik basıncı 2.5 bar") == ["lastik", "basıncı", "2", "5", "bar"]


def test_tokenize_works_with_empty_strings():
    assert tokenize("") == []


def test_stem_reduces_noun_suffixes():
    assert stem("ayarlar") == "ayar"
    assert stem("kapısını") == "kapı"
    assert stem("araçlar") == "araç"


def test_stem_lowercases_before_stemming():
    # The stemmer ignores uppercase suffixes, so stem() must lowercase first.
    assert stem("ARAÇLAR") == "araç"
    assert stem("Araçlar") == "araç"


def test_stem_leaves_unrecognised_verb_forms_unchanged():
    # Documents a known limitation of the Snowball Turkish stemmer.
    assert stem("ayarlanır") == "ayarlanır"
