# src/maintidoc/extract.py
import json
import pymupdf as fitz  # alias pour garder le reste du code inchangé
import pdfplumber
from pathlib import Path

from maintidoc.config import PDF_PATH, IMAGES_DIR, EXTRACTED


def extract_text_pages(pdf_path: str) -> list[dict]:
    """Retourne une liste de dicts {page, text} pour chaque page."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# src/maintidoc/extract.py
import pymupdf as fitz  # ou import fitz si vous gardez l'ancien nom

def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extrait les tableaux avec PyMuPDF find_tables().
    Essaie plusieurs stratégies pour maximiser la détection.
    """
    tables = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        # Essayer les stratégies dans l'ordre de pertinence pour les PDF LaTeX
        found_tables = []
        for strategy in ["lines", "lines_strict", "text"]:
            try:
                finder = page.find_tables(strategy=strategy)
                if finder.tables:
                    found_tables = finder.tables
                    break  # on garde la première stratégie qui trouve quelque chose
            except Exception as e:
                print(f"[warn] page {page_num} strategy={strategy}: {e}")
                continue

        for t_idx, table in enumerate(found_tables):
            try:
                md = table.to_markdown()
                if md and md.strip():
                    tables.append({
                        "page": page_num,
                        "table_index": t_idx,
                        "table_markdown": md,
                    })
            except Exception as e:
                print(f"[skip] table page {page_num} #{t_idx}: {e}")

    doc.close()
    return tables


def extract_images(pdf_path: str, output_dir: Path) -> list[dict]:
    """Sauvegarde chaque image en PNG et retourne {page, image_path}."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                filename = output_dir / f"page{page_num}_img{img_idx}.png"
                pix.save(filename)
                images.append({"page": page_num, "image_path": str(filename)})
                pix = None
            except Exception as e:
                print(f"[skip] image page {page_num} #{img_idx}: {e}")
    doc.close()
    return images


def describe_image_simple(image_meta: dict, page_text: str) -> str:
    """
    Description minimale : contexte textuel de la page + position.
    Évite un VLM lent ; suffit pour rendre l'image retrouvable.
    """
    snippet = page_text[:300].replace("\n", " ").strip()
    return (
        f"Image extraite de la page {image_meta['page']} du document technique. "
        f"Contexte de la page : {snippet}"
    )


def run_extraction() -> dict:
    """Pipeline complet d'extraction, retourne le dict et écrit le JSON."""
    print("→ Extraction du texte…")
    pages = extract_text_pages(str(PDF_PATH))
    print(f"   {len(pages)} pages avec texte")

    print("→ Extraction des tableaux…")
    tables = extract_tables(str(PDF_PATH))
    print(f"   {len(tables)} tableaux trouvés")

    print("→ Extraction des images…")
    images = extract_images(str(PDF_PATH), IMAGES_DIR)
    print(f"   {len(images)} images sauvegardées dans {IMAGES_DIR}")

    # Descriptions textuelles des images (pour indexation future)
    page_text_by_num = {p["page"]: p["text"] for p in pages}
    for img in images:
        img["description"] = describe_image_simple(
            img, page_text_by_num.get(img["page"], "")
        )

    result = {"pages": pages, "tables": tables, "images": images}

    with open(EXTRACTED, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"→ Résumé écrit dans {EXTRACTED}")

    return result


if __name__ == "__main__":
    run_extraction()