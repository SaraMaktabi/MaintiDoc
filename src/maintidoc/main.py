# src/maintidoc/main.py
"""
Point d'entrée CLI de MaintiDoc.

Usage :
    uv run python -m maintidoc.main extract
    uv run python -m maintidoc.main index      (à venir)
    uv run python -m maintidoc.main ask "..."  (à venir)
"""
import sys

from maintidoc.extract import run_extraction
from maintidoc.index import build_index
from maintidoc.rag import answer as rag_answer
from maintidoc.agent import answer as agent_answer
from maintidoc.index_llama import build_all as build_index_llama


def cmd_extract() -> None:
    result = run_extraction()
    print(
        f"\n✓ Extraction terminée : "
        f"{len(result['pages'])} pages, "
        f"{len(result['tables'])} tableaux, "
        f"{len(result['images'])} images."
    )


COMMANDS = {
    "extract": cmd_extract,
}


def cmd_index() -> None:
    build_index()


COMMANDS = {
    "extract": cmd_extract,
    "index": cmd_index,
}



def cmd_ask() -> None:
    if len(sys.argv) < 3:
        print('Usage : uv run python -m maintidoc.main ask "votre question"')
        sys.exit(1)
    question = " ".join(sys.argv[2:])
    result = rag_answer(question)
    print(f"\n>>> {question}\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, h in enumerate(result["sources"], start=1):
        print(
            f"[{i}] {h['content_type']} page {h['metadata'].get('page')} "
            f"(dist={h['distance']:.3f})"
        )


COMMANDS = {
    "extract": cmd_extract,
    "index": cmd_index,
    "ask": cmd_ask,
}

def cmd_agent() -> None:
    if len(sys.argv) < 3:
        print('Usage : uv run python -m maintidoc.main agent "votre question"')
        sys.exit(1)
    question = " ".join(sys.argv[2:])
    result = agent_answer(question)
    print(f"\n>>> {question}\n")
    print(result["answer"])
    if result.get("tool_calls"):
        print("\n=== Outils appelés ===")
        for tc in result["tool_calls"]:
            print(f"- {tc['name']}({tc['args']})")


COMMANDS = {
    "extract": cmd_extract,
    "index": cmd_index,
    "ask": cmd_ask,
    "agent": cmd_agent,
}


def cmd_index_llama() -> None:
    build_index_llama()


COMMANDS = {
    "extract": cmd_extract,
    "index": cmd_index,           # ancien, Chroma natif
    "index-llama": cmd_index_llama,  # nouveau, LlamaIndex
    "ask": cmd_ask,
    "agent": cmd_agent,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage : uv run python -m maintidoc.main <command>")
        print("Commandes disponibles :", ", ".join(COMMANDS))
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()  