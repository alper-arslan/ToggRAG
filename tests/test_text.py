from toggrag.text import turkish_lower


def test_dotted_and_dotless():
    assert turkish_lower("IŞIK") == "ışık"
    assert turkish_lower("İstanbul") == "istanbul"


def test_lower_and_empty_strings_dont_change():
    assert turkish_lower("ışık") == "ışık"
    assert turkish_lower("") == ""
