from toggrag.models import Chunk
from toggrag.pipeline import load_chunks, save_chunks


def test_chunks_survive_a_save_and_load(tmp_path):
    chunks = [
        Chunk(
            id=f"c{i}",
            text=f"Türkçe metin {i}\nikinci satır",
            pages=[i + 1],
            section_path=["1 Giriş"],
            content_type="text",
        )
        for i in range(3)
    ]
    path = tmp_path / "nested" / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
