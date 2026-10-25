from toggrag.text import analyze, clean_page_text, stem, tokenize, turkish_lower


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


def test_stem_leaves_unrecognised_verb_forms_unchanged():
    # Documents a known limitation of the Snowball Turkish stemmer.
    assert stem("ayarlanır") == "ayarlanır"


def test_analyze_tokenizes_and_stems():
    assert analyze("Lastikler ve KAPILAR") == ["lastik", "ve", "kapı"]


def test_analyze_empty_string_returns_empty_list():
    assert analyze("") == []


def test_removes_figure_ids():
    assert clean_page_text("Giriş\nTG02307\nMetin\nTG02816_T", 5) == "Giriş\nMetin"


def test_removes_callout_labels_after_a_figure_id():
    text = "TG02816_T\n2\n1\n3\n▪ Otomatik durdurmayı etkinleştirin."
    assert clean_page_text(text, 48) == "▪ Otomatik durdurmayı etkinleştirin."


def test_removes_the_printed_page_number_at_the_end():
    assert clean_page_text("Metin\n14", 14) == "Metin"


def test_keeps_numbers_that_belong_to_the_text():
    assert (
        clean_page_text("1. Akıllı ön kamera\nBasınç 2,5 bar", 9)
        == "1. Akıllı ön kamera\nBasınç 2,5 bar"
    )


def test_keeps_a_lone_number_that_does_not_follow_a_figure_id():
    assert clean_page_text("Boy\n2600\nmm", 9) == "Boy\n2600\nmm"


def test_keeps_lines_that_merely_mention_an_id():
    text = "Resim TG123 otomatik durdurma\nTG123 otomatik durdurma"
    assert clean_page_text(text, 9) == text
