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

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage : uv run python -m maintidoc.main <command>")
        print("Commandes disponibles :", ", ".join(COMMANDS))
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()  