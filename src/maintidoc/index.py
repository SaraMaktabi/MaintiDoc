# src/maintidoc/index.py
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from maintidoc.config import (
    EXTRACTED, CHROMA_DIR,
    COLLECTION_TEXT, COLLECTION_IMAGES,
    EMBED_MODEL_TEXT, CLIP_MODEL,
)


# ---------- Chunking ----------

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Chunking simple par caractères (≈512 tokens avec overlap)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if end >= len(text):
            break
    return [c.strip() for c in chunks if c.strip()]


# ---------- Construction des documents à indexer ----------

def build_text_documents(extracted: dict) -> list[dict]:
    """Prépare la liste des documents (id, text, metadata) pour la collection texte."""
    docs = []

    # 1. Pages de texte
    for p in extracted["pages"]:
        for i, chunk in enumerate(chunk_text(p["text"])):
            docs.append({
                "id": f"text_p{p['page']}_c{i}",
                "text": chunk,
                "metadata": {
                    "content_type": "text",
                    "page": p["page"],
                    "chunk_index": i,
                },
            })

    # 2. Tableaux (markdown)
    for t in extracted["tables"]:
        docs.append({
            "id": f"table_p{t['page']}_t{t['table_index']}",
            "text": t["table_markdown"],
            "metadata": {
                "content_type": "table",
                "page": t["page"],
                "table_index": t["table_index"],
            },
        })

    # 3. Descriptions d'images
    for j, img in enumerate(extracted["images"]):
        docs.append({
            "id": f"imgdesc_p{img['page']}_i{j}",
            "text": img["description"],
            "metadata": {
                "content_type": "image_description",
                "page": img["page"],
                "image_path": img["image_path"],
            },
        })

    return docs


# ---------- Indexation Chroma ----------

def index_text_collection(client, docs: list[dict]):
    """Indexe texte + tableaux + descriptions dans une collection Chroma."""
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_TEXT)

    # Reset propre si la collection existe déjà
    try:
        client.delete_collection(COLLECTION_TEXT)
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION_TEXT,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    coll.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"   → Collection '{COLLECTION_TEXT}' : {coll.count()} documents")
    return coll


def index_images_collection(client, images: list[dict]):
    """Encode les images avec CLIP et les stocke dans une collection dédiée."""
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    if not images:
        print("   → Aucune image à indexer (CLIP ignoré)")
        return None

    clip = SentenceTransformer(CLIP_MODEL)

    # Reset
    try:
        client.delete_collection(COLLECTION_IMAGES)
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION_IMAGES,
        metadata={"hnsw:space": "cosine"},
    )

    ids, embeddings, metadatas = [], [], []
    for j, img in enumerate(images):
        path = Path(img["image_path"])
        if not path.exists():
            continue
        pil = Image.open(path).convert("RGB")
        emb = clip.encode(pil, normalize_embeddings=True).tolist()
        ids.append(f"img_p{img['page']}_i{j}")
        embeddings.append(emb)
        metadatas.append({
            "page": img["page"],
            "image_path": img["image_path"],
            "description": img.get("description", ""),
        })

    if ids:
        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"   → Collection '{COLLECTION_IMAGES}' : {coll.count()} images")
    return coll


# ---------- Point d'entrée ----------

def build_index() -> None:
    print("→ Chargement de extracted.json…")
    with open(EXTRACTED, encoding="utf-8") as f:
        extracted = json.load(f)

    print("→ Connexion à Chroma (persistant)…")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print("→ Préparation des documents texte…")
    docs = build_text_documents(extracted)
    print(f"   {len(docs)} documents prêts (texte + tableaux + descriptions)")

    print("→ Indexation de la collection texte…")
    index_text_collection(client, docs)

    print("→ Indexation des images avec CLIP…")
    index_images_collection(client, extracted["images"])

    print("\n✓ Indexation terminée.")


if __name__ == "__main__":
    build_index()