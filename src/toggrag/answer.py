import re

from toggrag.llm import LLM
from toggrag.models import Answer, Chunk, Citation

NOT_FOUND = "Kılavuzda bu bilgiyi bulamadım."

SYSTEM_PROMPT = f"""Sen bir araç kullanım kılavuzu asistanısın. Soruyu YALNIZCA \
aşağıda numaralandırılmış kılavuz alıntılarına dayanarak Türkçe yanıtla.

Kurallar:
- Alıntılarda olmayan hiçbir bilgiyi ekleme, tahmin etme.
- Her bilgi cümlesinin sonuna dayandığın alıntının numarasını yaz, örn. [1].
- İşlem soruluyorsa adımları sırayla, numaralı liste olarak ver.
- Alıntılarda UYARI veya DİKKAT notu varsa yanıtta mutlaka belirt.
- Cevap alıntılarda yoksa yalnızca şunu yaz: {NOT_FOUND}"""

_CITATION = re.compile(r"\[(\d+)\]")


def format_context(chunks: list[Chunk]) -> str:
    """Number the chunks so the model can cite them."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        pages = ", ".join(str(page) for page in chunk.pages)
        section = " > ".join(chunk.section_path)
        parts.append(f"[{number}] (Sayfa {pages}; {section})\n{chunk.text}")
    return "\n\n".join(parts)


def parse_citations(text: str, chunks: list[Chunk]) -> list[Citation]:
    """Sources the answer cites, in order of first mention; unknown numbers are ignored."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= len(chunks) and number not in seen:
            seen.append(number)
    return [
        Citation(
            number=number,
            pages=chunks[number - 1].pages,
            section_path=chunks[number - 1].section_path,
        )
        for number in seen
    ]


def answer_question(question: str, chunks: list[Chunk], llm: LLM) -> Answer:
    """Answer from the retrieved chunks only; skip the LLM when nothing was found."""
    if not chunks:
        return Answer(text=NOT_FOUND, citations=[])
    user = f"Kılavuz alıntıları:\n\n{format_context(chunks)}\n\nSoru: {question}"
    text = llm.complete(SYSTEM_PROMPT, user).strip()
    if NOT_FOUND in text:
        return Answer(text=NOT_FOUND, citations=[])
    return Answer(text=text, citations=parse_citations(text, chunks))
