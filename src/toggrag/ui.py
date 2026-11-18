from toggrag.models import Answer


def format_answer(answer: Answer) -> str:
    """Answer text followed by a list of the manual pages it cites."""
    if not answer.citations:
        return answer.text
    sources = "\n".join(
        f"- [{c.number}] sayfa {', '.join(str(p) for p in c.pages)}: "
        f"{' > '.join(c.section_path)}"
        for c in answer.citations
    )
    return f"{answer.text}\n\n**Kaynaklar**\n{sources}"


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Start a minimal chat page. Needs the optional `ui` extra (gradio)."""
    import gradio as gr

    from toggrag.cli import build_pipeline

    pipeline = build_pipeline()

    def respond(message: str, history: list) -> str:
        return format_answer(pipeline.ask(message))

    gr.ChatInterface(
        respond,
        title="Togg Kullanım Kılavuzu Asistanı",
        description="Soruyu sorun; yanıt yalnızca kılavuza dayanır ve sayfa numarası verir.",
    ).launch(server_name=host, server_port=port)
