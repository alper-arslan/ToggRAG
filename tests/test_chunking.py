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
    assert all(b.page == 3 for b in blocks)


def test_split_blocks_starts_new_block_at_callout_header():
    blocks = split_blocks(1, "Metin devam\nDİKKAT\nDikkatli olun")
    assert [b.text for b in blocks] == ["Metin devam", "DİKKAT\nDikkatli olun"]


def test_split_blocks_keeps_all_sentences_of_a_list_item_together():
    blocks = split_blocks(1, "1. Aracı durdurun.\nMotoru tamamen kapatın.\n2. Çıkın.")
    assert [b.text for b in blocks] == [
        "1. Aracı durdurun.\nMotoru tamamen kapatın.",
        "2. Çıkın.",
    ]


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
    assert [(b.page, b.text) for b in alt.blocks] == [
        (2, "1.1 Alt"),
        (2, "Alt metni"),
        (3, "devam eden alt metin"),
    ]


def test_build_sections_removes_page_number_noise():
    sections = build_sections(_pages(), _headings())
    assert all(b.text != "3" for b in sections[-1].blocks)


def test_build_sections_without_headings_keeps_all_text_in_one_section():
    sections = build_sections(_pages(), [])
    assert len(sections) == 1
    assert sections[0].path == []
    assert any("Giriş metni." in b.text for b in sections[0].blocks)
    assert sections[0].blocks[0].page == 1


def test_build_sections_keeps_heading_that_is_not_found_in_text():
    pages = [Page(page=1, text="Başka şeyler")]
    sections = build_sections(pages, [Heading(level=1, title="Yok", page=1)])
    assert [b.text for b in sections[0].blocks] == ["Yok", "Başka şeyler"]


def _section(path, *words):
    return Section(path, [Block(1, " ".join(["kelime"] * n)) for n in words])


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
    return [Block(1, " ".join([f"k{i}"] * n)) for i, n in enumerate(sizes)]


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
        Block(1, "Metin devam."),
        Block(1, "DİKKAT"),
        Block(1, "▪ Kemer takın."),
        Block(1, "Normal paragraf."),
    ]
    mark_callouts(blocks)
    assert [b.callout for b in blocks] == [False, True, True, False]


def test_classify_warning():
    blocks = [Block(1, "DİKKAT"), Block(1, "▪ Kemer takın."), Block(1, "▪ Yavaşlayın.")]
    mark_callouts(blocks)
    assert classify(blocks) == "warning"


def test_classify_warning_chunk_that_starts_after_the_callout_header():
    blocks = [Block(1, "▪ Kemer takın."), Block(1, "▪ Yavaşlayın.")]
    assert classify(blocks) == "procedure"
    for block in blocks:
        block.callout = True
    assert classify(blocks) == "warning"


def test_classify_procedure():
    blocks = [Block(1, "1. Kapıyı açın."), Block(1, "2. Anahtarı çevirin.")]
    assert classify(blocks) == "procedure"


def test_classify_specification():
    assert classify([Block(1, "Boy 4515 mm\nEn 1863 mm\n1630 2600")]) == "specification"


def test_classify_plain_text():
    assert classify([Block(1, "Araç sessiz çalışır.")]) == "text"


def test_chunk_document_builds_chunks_with_pages_and_path():
    chunks = chunk_document(_pages(), _headings(), ChunkingConfig(target_tokens=100))
    assert [c.id for c in chunks] == ["chunk-0000", "chunk-0001"]
    assert chunks[0].section_path == ["1 Giriş"]
    assert chunks[0].pages == [2, 3]
    assert chunks[1].section_path == ["2 İkinci"]


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
