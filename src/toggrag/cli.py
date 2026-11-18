import argparse

from toggrag.config import load_settings
from toggrag.embeddings import make_embedder
from toggrag.llm import make_llm
from toggrag.pipeline import RagPipeline, ingest, load_retriever


def build_pipeline() -> RagPipeline:
    settings = load_settings()
    retriever = load_retriever(settings, make_embedder(settings.embedding))
    return RagPipeline(retriever, make_llm(settings))


def _ingest(_: argparse.Namespace) -> None:
    settings = load_settings()
    chunks = ingest(settings, make_embedder(settings.embedding))
    print(f"{len(chunks)} chunks written to {settings.data_dir}")


def _ask(args: argparse.Namespace) -> None:
    answer = build_pipeline().ask(args.question, args.mode)
    print(answer.text)
    for citation in answer.citations:
        pages = ", ".join(str(page) for page in citation.pages)
        print(
            f"[{citation.number}] sayfa {pages} - {' > '.join(citation.section_path)}"
        )


def _eval(args: argparse.Namespace) -> None:
    from toggrag.evaluation import run_eval

    run_eval(args.questions, args.faithfulness)


def main() -> None:
    parser = argparse.ArgumentParser(prog="toggrag")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "ingest", help="parse, chunk, embed and index the manual"
    ).set_defaults(run=_ingest)

    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ask.set_defaults(run=_ask)

    evaluate = commands.add_parser("eval", help="compare retrieval configurations")
    evaluate.add_argument("--questions", default="eval/questions.yaml")
    evaluate.add_argument(
        "--faithfulness", action="store_true", help="also judge answers with the LLM"
    )
    evaluate.set_defaults(run=_eval)

    args = parser.parse_args()
    args.run(args)
