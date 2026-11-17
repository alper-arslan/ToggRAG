import pytest

from toggrag.chunking import (
    Block,
    Section,
    build_sections,
    chunk_document,
    classify,
    contextual_text,
    find_heading,
    mark_callouts,
    merge_sections,
    pack_blocks,
    split_blocks,
    split_oversized_blocks,
)
from toggrag.config import ChunkingConfig
from toggrag.models import Chunk, Heading, Page


def test_find_heading_matches_a_wrapped_title():
    text = "Giriş\n3.1.3 Kaybolması/\nDeğiştirilmesi\nMetin"
    match = find_heading("3.1.3 Kaybolması/Değiştirilmesi", text)
    assert match is not None
    assert text[match.end() :] == "\nMetin"


def test_find_heading_ignores_titles_inside_a_line():
    assert find_heading("2.1 Güvenlik", "bkz. 2.1 Güvenlik") is None


def test_find_heading_does_not_match_a_longer_title():
    text = "2.1 Güvenlik Talimatları\nMetin\n2.1 Güvenlik\nSon"
    match = find_heading("2.1 Güvenlik", text)
    assert match is not None
    assert text[match.end() :] == "\nSon"


def test_find_heading_searches_from_start():
    text = "2.1 Güvenlik\nMetin\n2.1 Güvenlik"
    match = find_heading("2.1 Güvenlik", text, start=1)
    assert match is not None and match.start() == len(text) - len("2.1 Güvenlik")


def test_split_blocks_starts_new_block_at_list_items():
    blocks = split_blocks(
        3, "Giriş cümlesi\ndevam eder\n1. Birinci adım\n2. İkinci adım"
    )
    assert [b.text for b in blocks] == [
        "Giriş cümlesi\ndevam eder",
        "1. Birinci adım",
        "2. İkinci adım",
    ]
    assert all(b.pages == [3] for b in blocks)


def test_split_blocks_starts_new_block_at_callout_header():
    blocks = split_blocks(1, "Metin devam\nDİKKAT\nDikkatli olun")
    assert [b.text for b in blocks] == ["Metin devam", "DİKKAT\nDikkatli olun"]


def test_split_blocks_keeps_all_sentences_of_a_list_item_together():
    blocks = split_blocks(1, "1. Aracı durdurun.\nMotoru tamamen kapatın.\n2. Çıkın.")
    assert [b.text for b in blocks] == [
        "1. Aracı durdurun.\nMotoru tamamen kapatın.",
        "2. Çıkın.",
    ]


def test_split_blocks_short_all_caps_units_do_not_start_a_block():
    assert len(split_blocks(1, "Hız 50\nKM\nile sınırlı")) == 1
    assert len(split_blocks(1, "Metin\nNOT\nBir şey")) == 2


def test_split_blocks_splits_after_sentence_end_only():
    blocks = split_blocks(1, "Birinci cümle.\nİkinci cümle\ndevam eder")
    assert [b.text for b in blocks] == ["Birinci cümle.", "İkinci cümle\ndevam eder"]


def test_split_blocks_empty_text():
    assert split_blocks(1, "\n  \n") == []


def _pages():
    return [
        Page(page=1, text="Kapak"),
        Page(page=2, text="1 Giriş\nGiriş metni.\n1.1 Alt\nAlt metni"),
        Page(page=3, text="devam eden alt metin\n2 İkinci\nBaşka metin\n3"),
    ]


def _headings():
    return [
        Heading(level=1, title="1 Giriş", page=2),
        Heading(level=2, title="1.1 Alt", page=2),
        Heading(level=1, title="2 İkinci", page=3),
    ]


def test_build_sections_drops_text_before_the_first_heading():
    sections = build_sections(_pages(), _headings())
    assert all("Kapak" not in b.text for s in sections for b in s.blocks)


def test_build_sections_sets_path_and_pages():
    sections = build_sections(_pages(), _headings())
    assert [s.path for s in sections] == [
        ["1 Giriş"],
        ["1 Giriş", "1.1 Alt"],
        ["2 İkinci"],
    ]
    alt = sections[1]
    assert [(b.pages, b.text) for b in alt.blocks] == [
        ([2], "1.1 Alt"),
        ([2, 3], "Alt metni\ndevam eden alt metin"),
    ]


def test_build_sections_removes_page_number_noise():
    sections = build_sections(_pages(), _headings())
    assert all(b.text != "3" for b in sections[-1].blocks)


def test_build_sections_without_headings_keeps_all_text_in_one_section():
    sections = build_sections(_pages(), [])
    assert len(sections) == 1
    assert sections[0].path == []
    assert any("Giriş metni." in b.text for b in sections[0].blocks)
    assert 1 in sections[0].blocks[0].pages


def test_build_sections_joins_a_list_item_that_runs_over_the_page_break():
    pages = [
        Page(page=2, text="1 Giriş\nDİKKAT\n▪ Kemer takın ve"),
        Page(page=3, text="bağlantısını kontrol edin.\n▪ Yavaşlayın."),
    ]
    sections = build_sections(pages, [Heading(level=1, title="1 Giriş", page=2)])
    blocks = sections[0].blocks
    assert [(b.pages, b.text) for b in blocks] == [
        ([2], "1 Giriş"),
        ([2], "DİKKAT"),
        ([2, 3], "▪ Kemer takın ve\nbağlantısını kontrol edin."),
        ([3], "▪ Yavaşlayın."),
    ]
    mark_callouts(blocks)
    assert [b.callout for b in blocks] == [False, True, True, True]


def test_build_sections_does_not_glue_a_capitalised_paragraph_onto_an_item():
    pages = [
        Page(page=2, text="1 Giriş\n▪ Aracı durdurun"),
        Page(page=3, text="Kontak açın"),
    ]
    sections = build_sections(pages, [Heading(level=1, title="1 Giriş", page=2)])
    assert [b.text for b in sections[0].blocks] == [
        "1 Giriş",
        "▪ Aracı durdurun",
        "Kontak açın",
    ]


def test_build_sections_raises_for_a_heading_on_a_missing_page():
    pages = [Page(page=1, text="A"), Page(page=3, text="C")]
    headings = [Heading(level=1, title="B", page=2)]
    with pytest.raises(ValueError, match=r"\[2\]"):
        build_sections(pages, headings)


def test_build_sections_marks_only_the_heading_block():
    alt = build_sections(_pages(), _headings())[1]
    assert alt.blocks[0].heading
    assert not any(b.heading for b in alt.blocks[1:])


def test_build_sections_does_not_join_after_a_finished_sentence():
    pages = [Page(page=2, text="1 Giriş\nBitti."), Page(page=3, text="Yeni cümle")]
    sections = build_sections(pages, [Heading(level=1, title="1 Giriş", page=2)])
    assert [b.text for b in sections[0].blocks] == ["1 Giriş", "Bitti.", "Yeni cümle"]


def test_build_sections_never_joins_text_into_the_heading_line():
    pages = [Page(page=2, text="1 Giriş"), Page(page=3, text="metin devam eder")]
    sections = build_sections(pages, [Heading(level=1, title="1 Giriş", page=2)])
    assert [b.text for b in sections[0].blocks] == ["1 Giriş", "metin devam eder"]


def test_build_sections_without_headings_joins_across_pages_too():
    pages = [Page(page=1, text="1. Adım bir ve"), Page(page=2, text="devamı.")]
    blocks = build_sections(pages, [])[0].blocks
    assert [(b.pages, b.text) for b in blocks] == [([1, 2], "1. Adım bir ve\ndevamı.")]


def test_build_sections_joins_across_one_page_break_at_most():
    pages = [Page(page=n, text=f"satır {n} devam") for n in (1, 2, 3)]
    blocks = build_sections(pages, [])[0].blocks
    assert [b.pages for b in blocks] == [[1, 2], [3]]


def test_build_sections_keeps_heading_that_is_not_found_in_text():
    pages = [Page(page=1, text="Başka şeyler")]
    sections = build_sections(pages, [Heading(level=1, title="Yok", page=1)])
    assert [b.text for b in sections[0].blocks] == ["Yok", "Başka şeyler"]


def _section(path, *words):
    return Section(path, [Block([1], " ".join(["kelime"] * n)) for n in words])


def test_merge_sections_joins_small_sections_of_the_same_chapter():
    sections = [_section(["1", "1.1"], 5), _section(["1", "1.2"], 5)]
    merged = merge_sections(sections, target_tokens=20)
    assert len(merged) == 1
    assert merged[0][0] == ["1"]


def test_merge_sections_keeps_full_path_of_a_single_section():
    merged = merge_sections([_section(["1", "1.1"], 5)], target_tokens=20)
    assert merged[0][0] == ["1", "1.1"]


def test_merge_sections_does_not_cross_chapters():
    sections = [_section(["1"], 5), _section(["2"], 5)]
    assert len(merge_sections(sections, target_tokens=20)) == 2


def test_merge_sections_respects_target_size():
    sections = [_section(["1", "1.1"], 15), _section(["1", "1.2"], 15)]
    assert len(merge_sections(sections, target_tokens=20)) == 2


def _blocks(*sizes):
    return [Block([1], " ".join([f"k{i}"] * n)) for i, n in enumerate(sizes)]


def test_pack_blocks_never_splits_a_block():
    blocks = _blocks(8, 8, 8)
    chunks = pack_blocks(blocks, target_tokens=10, overlap_tokens=0)
    assert chunks == [[blocks[0]], [blocks[1]], [blocks[2]]]


def test_pack_blocks_keeps_everything_in_one_chunk_when_it_fits():
    blocks = _blocks(3, 3, 3)
    assert pack_blocks(blocks, target_tokens=10, overlap_tokens=0) == [blocks]


def test_pack_blocks_repeats_tail_as_overlap():
    blocks = _blocks(4, 4, 4)
    chunks = pack_blocks(blocks, target_tokens=10, overlap_tokens=4)
    assert chunks == [[blocks[0], blocks[1]], [blocks[1], blocks[2]]]


def test_pack_blocks_overlap_never_pushes_a_chunk_past_the_target():
    blocks = _blocks(4, 4, 8)
    chunks = pack_blocks(blocks, target_tokens=10, overlap_tokens=4)
    sizes = [sum(len(b.text.split()) for b in chunk) for chunk in chunks]
    assert all(size <= 10 for size in sizes)


def test_mark_callouts_flags_header_and_items_under_it():
    blocks = [
        Block([1], "Metin devam."),
        Block([1], "DİKKAT"),
        Block([1], "▪ Kemer takın."),
        Block([1], "Normal paragraf."),
    ]
    mark_callouts(blocks)
    assert [b.callout for b in blocks] == [False, True, True, False]


def test_pack_blocks_keeps_a_heading_with_the_content_after_it():
    heading = Block([1], "Başlık", heading=True)
    big = Block([1], " ".join(["kelime"] * 50))
    assert pack_blocks([heading, big], target_tokens=20, overlap_tokens=0) == [
        [heading, big]
    ]


def test_split_oversized_blocks_cuts_at_line_boundaries():
    text = "\n".join(["bir iki"] * 6)
    block = Block([4, 5], text, callout=True)
    pieces = split_oversized_blocks([block], target_tokens=5)
    assert [p.text for p in pieces] == ["\n".join(["bir iki"] * 2)] * 3
    assert all(p.pages == [4, 5] and p.callout for p in pieces)


def test_split_oversized_blocks_cites_only_the_pages_of_each_piece():
    block = Block(
        [4, 5],
        "\n".join(["bir iki"] * 4),
        line_pages=[4, 4, 5, 5],
    )
    pieces = split_oversized_blocks([block], target_tokens=4)
    assert [p.pages for p in pieces] == [[4], [5]]


def test_split_oversized_blocks_keeps_all_pages_without_line_pages():
    block = Block([4, 5], "\n".join(["bir iki"] * 4))
    pieces = split_oversized_blocks([block], target_tokens=4)
    assert [p.pages for p in pieces] == [[4, 5], [4, 5]]


def test_split_oversized_blocks_slices_line_pages_and_spans_a_page_break():
    block = Block([4, 5], "\n".join(["bir iki"] * 4), line_pages=[4, 5, 5, 5])
    pieces = split_oversized_blocks([block], target_tokens=4)
    assert [p.pages for p in pieces] == [[4, 5], [5]]
    assert [p.line_pages for p in pieces] == [[4, 5], [5, 5]]


def test_build_sections_records_the_page_of_every_line_of_a_joined_item():
    pages = [
        Page(page=1, text="▪ Birinci adım\nikinci satır"),
        Page(page=2, text="devam eder."),
    ]
    (section,) = build_sections(pages, [])
    assert section.blocks[0].pages == [1, 2]
    assert section.blocks[0].line_pages == [1, 1, 2]


def test_split_oversized_blocks_keeps_a_long_heading_whole():
    heading = Block([1], "çok uzun bir başlık satırı burada", heading=True)
    (result,) = split_oversized_blocks([heading], target_tokens=3)
    assert result is heading


def test_split_oversized_blocks_leaves_small_blocks_and_long_lines_alone():
    small = Block([1], "bir iki")
    long_line = Block([1], " ".join(["kelime"] * 9))
    assert split_oversized_blocks([small, long_line], target_tokens=5) == [
        small,
        long_line,
    ]


def test_chunk_document_splits_a_huge_unpunctuated_block():
    lines = "\n".join(f"madde {n}" for n in range(30))
    pages = [Page(page=2, text=f"1 Giriş\n{lines}")]
    chunks = chunk_document(
        pages,
        [Heading(level=1, title="1 Giriş", page=2)],
        ChunkingConfig(target_tokens=10, overlap_ratio=0),
    )
    sizes = [len(c.text.split()) for c in chunks]
    assert len(chunks) > 3
    assert chunks[0].text.startswith("1 Giriş")
    assert sizes[0] <= 10 + 2  # the heading travels with the first piece
    assert all(size <= 10 for size in sizes[1:])


def test_classify_ignores_list_numbering_in_both_counts():
    plain = [Block([1], "aaa 5 bb\nccc 5 dd")]
    numbered = [Block([1], "1. aaa 5 bb\n2. ccc 5 dd")]
    assert classify(plain) == classify(numbered) == "specification"


def test_classify_warning():
    blocks = [
        Block([1], "DİKKAT"),
        Block([1], "▪ Kemer takın."),
        Block([1], "▪ Yavaşlayın."),
    ]
    mark_callouts(blocks)
    assert classify(blocks) == "warning"


def test_classify_warning_chunk_that_starts_after_the_callout_header():
    blocks = [Block([1], "▪ Kemer takın."), Block([1], "▪ Yavaşlayın.")]
    assert classify(blocks) == "procedure"
    for block in blocks:
        block.callout = True
    assert classify(blocks) == "warning"


def test_classify_procedure():
    blocks = [Block([1], "1. Kapıyı açın."), Block([1], "2. Anahtarı çevirin.")]
    assert classify(blocks) == "procedure"


def test_classify_specification():
    assert (
        classify([Block([1], "Boy 4515 mm\nEn 1863 mm\n1630 2600")]) == "specification"
    )


def test_classify_plain_text():
    assert classify([Block([1], "Araç sessiz çalışır.")]) == "text"


def test_chunk_document_builds_chunks_with_pages_and_path():
    chunks = chunk_document(_pages(), _headings(), ChunkingConfig(target_tokens=100))
    assert [c.id for c in chunks] == ["chunk-0000", "chunk-0001"]
    assert chunks[0].section_path == ["1 Giriş"]
    assert chunks[0].pages == [2, 3]
    assert chunks[1].section_path == ["2 İkinci"]


def test_chunk_document_skips_a_section_with_only_a_heading():
    pages = [Page(page=2, text="1 Giriş\nMetin burada.\n2 Boş")]
    headings = [
        Heading(level=1, title="1 Giriş", page=2),
        Heading(level=1, title="2 Boş", page=2),
    ]
    chunks = chunk_document(pages, headings, ChunkingConfig(target_tokens=3))
    assert all("Boş" not in c.text for c in chunks)
    assert chunks


def test_chunk_document_without_headings_still_returns_chunks():
    chunks = chunk_document(_pages(), [], ChunkingConfig(target_tokens=100))
    assert chunks
    assert all(c.section_path == [] for c in chunks)
    assert contextual_text(chunks[0]) == chunks[0].text


def test_contextual_text_puts_section_path_first():
    chunk = Chunk(
        id="c",
        text="Metin",
        pages=[1],
        section_path=["1 Giriş", "1.1 Alt"],
        content_type="text",
    )
    assert contextual_text(chunk) == "1 Giriş > 1.1 Alt\nMetin"


def test_build_sections_siblings_after_a_level_jump_share_their_parent():
    pages = [Page(page=1, text="Kök\nİlk\nMetin bir.\nİkinci\nMetin iki.")]
    headings = [
        Heading(level=1, title="Kök", page=1),
        Heading(level=3, title="İlk", page=1),
        Heading(level=3, title="İkinci", page=1),
    ]
    sections = build_sections(pages, headings)
    assert [s.path for s in sections] == [
        ["Kök"],
        ["Kök", "İlk"],
        ["Kök", "İkinci"],
    ]


def test_mark_callouts_ends_at_a_numbered_heading():
    blocks = [
        Block([1], "DİKKAT"),
        Block([1], "▪ Kemer takın."),
        Block([1], "2. Kullanım", heading=True),
        Block([1], "1. Kapıyı açın."),
    ]
    mark_callouts(blocks)
    assert [b.callout for b in blocks] == [True, True, False, False]


def test_pack_blocks_moves_a_trailing_heading_to_the_next_chunk():
    text = Block([1], " ".join(["a"] * 8))
    heading = Block([1], "Başlık", heading=True)
    big = Block([1], " ".join(["b"] * 12))
    chunks = pack_blocks([text, heading, big], target_tokens=16, overlap_tokens=0)
    assert chunks == [[text], [heading, big]]


def test_classify_ignores_the_numbers_of_headings():
    blocks = [Block([1], "3.1.2 Teknik", heading=True), Block([1], "Fren bakımı.")]
    assert classify(blocks) == "text"


def test_build_sections_keeps_the_hierarchy_when_the_outline_order_differs():
    pages = [Page(page=1, text="Bölüm\nmetin."), Page(page=2, text="Kök\nmetin.")]
    headings = [
        Heading(level=1, title="Kök", page=2),
        Heading(level=2, title="Bölüm", page=1),
    ]
    assert [s.path for s in build_sections(pages, headings)] == [
        ["Kök", "Bölüm"],
        ["Kök"],
    ]


def test_build_sections_keeps_text_before_a_missing_heading_with_it():
    pages = [Page(page=1, text="alpha prose.\nPresent\nbeta prose.")]
    headings = [
        Heading(level=1, title="Absent", page=1),
        Heading(level=1, title="Present", page=1),
    ]
    absent, present = build_sections(pages, headings)
    assert [b.text for b in absent.blocks] == ["Absent", "alpha prose."]
    assert [b.text for b in present.blocks] == ["Present", "beta prose."]


def test_pack_blocks_puts_a_carried_heading_after_the_overlap():
    old, small = Block([1], " ".join(["a"] * 12)), Block([1], "x y")
    heading = Block([1], "Başlık", heading=True)
    new = Block([1], " ".join(["b"] * 12))
    chunks = pack_blocks([old, small, heading, new], target_tokens=16, overlap_tokens=2)
    assert chunks[1] == [small, heading, new]


def test_build_sections_logs_a_heading_missing_from_the_text(caplog):
    pages = [Page(page=1, text="A\nmetin.")]
    headings = [
        Heading(level=1, title="A", page=1),
        Heading(level=1, title="Z", page=1),
    ]
    with caplog.at_level("WARNING"):
        build_sections(pages, headings)
    assert "heading not found" in caplog.text


def test_build_sections_rejects_unordered_pages():
    pages = [Page(page=2, text="x"), Page(page=1, text="y")]
    with pytest.raises(ValueError, match="increasing"):
        build_sections(pages, [])


def test_split_blocks_keeps_an_all_caps_word_inside_a_list_item():
    blocks = split_blocks(1, "1. Düğmeye basın ve\nLED\nyanana kadar bekleyin.")
    assert len(blocks) == 1
    assert [b.text for b in split_blocks(1, "1. Basın\nDİKKAT\nSıcak.")][
        1
    ] == "DİKKAT\nSıcak."


def test_a_year_or_street_number_is_not_a_list_item():
    assert len(split_blocks(1, "bu iş\n2024. yılında bitti.")) == 1


def test_build_sections_gives_repeated_titles_on_one_page_their_own_text():
    pages = [Page(page=1, text="1 Ön\nÖzet\nbir.\n2 Arka\nÖzet\niki.")]
    headings = [
        Heading(level=1, title="1 Ön", page=1),
        Heading(level=2, title="Özet", page=1),
        Heading(level=1, title="2 Arka", page=1),
        Heading(level=2, title="Özet", page=1),
    ]
    sections = build_sections(pages, headings)
    assert [s.path for s in sections][1::2] == [["1 Ön", "Özet"], ["2 Arka", "Özet"]]
    assert [b.text for b in sections[1].blocks] == ["Özet", "bir."]
    assert [b.text for b in sections[3].blocks] == ["Özet", "iki."]
