import re
from collections import defaultdict
from dataclasses import dataclass, field

from toggrag.config import ChunkingConfig
from toggrag.models import Chunk, Heading, Page
from toggrag.text import clean_page_text, tokenize

_ITEM = re.compile(r"^(?:[▪•●■\-–]|\d+[.)])\s")
_SENTENCE_END = (".", ":", "!", "?")
_CALLOUTS = {"UYARI", "DİKKAT", "TEHLİKE", "İKAZ"}
_MAX_HEADER_LENGTH = 30


@dataclass
class Block:
    """A run of lines that stay together: a list item, a callout header, a paragraph.

    A block that runs over a page break lists every page it touches.
    """

    pages: list[int]
    text: str
    callout: bool = False


@dataclass
class Section:
    """The blocks that belong to one heading of the outline."""

    path: list[str]
    blocks: list[Block] = field(default_factory=list)


def find_heading(title: str, text: str, start: int = 0) -> re.Match[str] | None:
    """Find a heading that fills its line(s), even if the PDF wrapped it.

    A longer title that only starts with the same words ("2.1 Güvenlik Talimatları"
    for "2.1 Güvenlik") does not match. Whitespace inside the title is ignored, so
    "3.1.3 Kaybolması/Değiştirilmesi" also matches "3.1.3 Kaybolması/\\nDeğiştirilmesi".
    """
    letters = [re.escape(char) for char in title if not char.isspace()]
    pattern = re.compile(r"^" + r"\s*".join(letters) + r"[ \t]*$", re.MULTILINE)
    return pattern.search(text, start)


def _starts_block(line: str, current: list[str]) -> bool:
    if _ITEM.match(line) or (line.isupper() and len(line) <= _MAX_HEADER_LENGTH):
        return True
    if _ITEM.match(current[0]):
        return False
    return current[-1].endswith(_SENTENCE_END) and line[0].isupper()


def split_blocks(page: int, text: str) -> list[Block]:
    """Group the hard-wrapped lines of a page into blocks.

    The PDF has no blank lines between paragraphs, so a new block starts at a
    list item, at an all-caps header such as DİKKAT, and, outside list items,
    after a line that ends a sentence. A list item keeps all its sentences.
    Chunks are only cut between blocks, so an item is never cut in half.
    """
    blocks: list[Block] = []
    current: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if current and _starts_block(line, current):
            blocks.append(Block([page], "\n".join(current)))
            current = []
        current.append(line)
    if current:
        blocks.append(Block([page], "\n".join(current)))
    return blocks


def _continues_across_pages(first_line: str, last: Block) -> bool:
    """Does the next page's first line continue the previous page's last block?

    Only when the block was cut mid-sentence and the line is not a new item or
    header, and either the block is a list item or the line starts in lowercase.
    A block is joined across one page break at most, so lines without sentence
    ends (tables, indexes) cannot chain into one huge block.
    """
    lines = last.text.splitlines()
    if len(last.pages) > 1 or lines[-1].endswith(_SENTENCE_END):
        return False
    if _starts_block(first_line, lines):
        return False
    return bool(_ITEM.match(lines[0])) or first_line[0].islower()


def _add_text(section: Section, page: int, text: str) -> None:
    """Append a page's text to a section.

    If the page starts in the middle of the section's last block (a list item or
    paragraph that runs over the page break), the two are joined so the item is
    not cut and keeps all its pages. A section's heading line is never joined.
    """
    blocks = split_blocks(page, text)
    can_join = section.blocks and (len(section.blocks) > 1 or not section.path)
    if blocks and can_join:
        last, first = section.blocks[-1], blocks[0]
        if _continues_across_pages(first.text.splitlines()[0], last):
            last.text += "\n" + first.text
            last.pages.append(page)
            blocks = blocks[1:]
    section.blocks += blocks


def build_sections(pages: list[Page], headings: list[Heading]) -> list[Section]:
    """Cut the document at its outline headings.

    A heading that cannot be found in the page text still opens a section at the
    current position. Text before the first heading (cover, table of contents)
    belongs to no section and is dropped. Without any headings (a PDF with no
    bookmarks) the whole document becomes one section with an empty path.
    """
    if not headings:
        section = Section([])
        for page in pages:
            _add_text(section, page.page, clean_page_text(page.text, page.page))
        return [section]

    by_page: dict[int, list[Heading]] = defaultdict(list)
    for heading in headings:
        by_page[heading.page].append(heading)

    sections: list[Section] = []
    path: list[str] = []
    current: Section | None = None
    for page in pages:
        text = clean_page_text(page.text, page.page)
        pos = 0
        for heading in by_page[page.page]:
            match = find_heading(heading.title, text, pos)
            start = match.start() if match else pos
            if current:
                _add_text(current, page.page, text[pos:start])
            path = path[: heading.level - 1] + [heading.title]
            current = Section(path, [Block([page.page], heading.title)])
            sections.append(current)
            pos = match.end() if match else pos
        if current:
            _add_text(current, page.page, text[pos:])
    return sections


def _tokens(blocks: list[Block]) -> int:
    return sum(len(tokenize(block.text)) for block in blocks)


def _common_prefix(paths: list[list[str]]) -> list[str]:
    prefix = paths[0]
    for path in paths[1:]:
        shared = 0
        while shared < min(len(prefix), len(path)) and prefix[shared] == path[shared]:
            shared += 1
        prefix = prefix[:shared]
    return prefix


def merge_sections(
    sections: list[Section], target_tokens: int
) -> list[tuple[list[str], list[Block]]]:
    """Join neighbouring small sections of the same chapter up to the target size.

    The merged group keeps the headings it covers as text and takes the part of
    the section path they share.
    """
    groups: list[tuple[list[list[str]], list[Block]]] = []
    for section in sections:
        if groups:
            paths, blocks = groups[-1]
            same_chapter = paths[0][:1] == section.path[:1]
            if same_chapter and _tokens(blocks + section.blocks) <= target_tokens:
                paths.append(section.path)
                blocks += section.blocks
                continue
        groups.append(([section.path], list(section.blocks)))
    return [(_common_prefix(paths), blocks) for paths, blocks in groups]


def pack_blocks(
    blocks: list[Block], target_tokens: int, overlap_tokens: int
) -> list[list[Block]]:
    """Fill chunks with whole blocks up to the target size.

    When a chunk is full the next one starts with the last blocks of the
    previous one (up to overlap_tokens) so a step at the border is not lost.
    The overlap is cut back so it never pushes the chunk past the target; a
    single block bigger than the target stays whole.
    """
    chunks: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        n = _tokens([block])
        if current and size + n > target_tokens:
            chunks.append(current)
            current = _overlap_tail(current, min(overlap_tokens, target_tokens - n))
            size = _tokens(current)
        current.append(block)
        size += n
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(blocks: list[Block], overlap_tokens: int) -> list[Block]:
    tail: list[Block] = []
    size = 0
    for block in reversed(blocks):
        size += _tokens([block])
        if size > overlap_tokens:
            break
        tail.insert(0, block)
    return tail


def mark_callouts(blocks: list[Block]) -> None:
    """Flag the blocks that belong to a callout: the header and the items under it.

    Done on the whole section before packing, so a chunk that starts in the
    middle of a callout still knows it is a warning.
    """
    in_callout = False
    for block in blocks:
        first_line = block.text.splitlines()[0]
        if first_line in _CALLOUTS:
            in_callout = True
        elif not _ITEM.match(first_line):
            in_callout = False
        block.callout = in_callout


def classify(blocks: list[Block]) -> str:
    """Guess what kind of content a chunk holds.

    warning: mostly callout text (see mark_callouts);
    specification: mostly numbers (list numbering excluded); procedure: two or more list items;
    text: everything else.
    """
    total = _tokens(blocks) or 1
    callout_tokens = _tokens([block for block in blocks if block.callout])
    items = sum(
        bool(_ITEM.match(line)) for block in blocks for line in block.text.splitlines()
    )
    words = [
        word
        for block in blocks
        for line in block.text.splitlines()
        for word in tokenize(_ITEM.sub("", line))
    ]
    if callout_tokens / total >= 0.5:
        return "warning"
    if sum(word.isdigit() for word in words) / total >= 0.3:
        return "specification"
    if items >= 2:
        return "procedure"
    return "text"


def chunk_document(
    pages: list[Page], headings: list[Heading], config: ChunkingConfig
) -> list[Chunk]:
    """Turn the manual into chunks that carry their section path and pages."""
    overlap_tokens = int(config.target_tokens * config.overlap_ratio)
    chunks: list[Chunk] = []
    sections = build_sections(pages, headings)
    for path, blocks in merge_sections(sections, config.target_tokens):
        mark_callouts(blocks)
        for group in pack_blocks(blocks, config.target_tokens, overlap_tokens):
            chunks.append(
                Chunk(
                    id=f"chunk-{len(chunks):04d}",
                    text="\n".join(block.text for block in group),
                    pages=sorted({page for block in group for page in block.pages}),
                    section_path=path,
                    content_type=classify(group),
                )
            )
    return chunks


def contextual_text(chunk: Chunk) -> str:
    """Text to embed and index: the section path in front of the chunk text."""
    if not chunk.section_path:
        return chunk.text
    return " > ".join(chunk.section_path) + "\n" + chunk.text
