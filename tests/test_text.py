from toggrag.text import tokenize, turkish_lower


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
