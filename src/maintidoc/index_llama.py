# src/maintidoc/index_llama.py
"""
Indexation via LlamaIndex.
Texte + tableaux + descriptions d'images -> ChromaVectorStore (collection texte)
Images -> embeddings CLIP manuels (collection images)
"""
import json

import chromadb
from llama_index.core import (
    VectorStoreIndex, StorageContext, Document, Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from maintidoc.config import (
    EXTRACTED, CHROMA_DIR,
    COLLECTION_TEXT, COLLECTION_IMAGES,
    EMBED_MODEL_TEXT, CLIP_MODEL,
)


def build_documents(extracted: dict) -> list[Document]:
    """Convertit extracted.json en liste de Documents LlamaIndex."""
    docs = []

    # Texte (un Document par page, LlamaIndex chunkera ensuite)
    for p in extracted["pages"]:
        docs.append(Document(
            text=p["text"],
            metadata={"content_type": "text", "page": p["page"]},
        ))

    # Tableaux (un Document par tableau)
    for t in extracted["tables"]:
        docs.append(Document(
            text=t["table_markdown"],
            metadata={
                "content_type": "table",
                "page": t["page"],
                "table_index": t["table_index"],
            },
        ))

    # Descriptions d'images
    for j, img in enumerate(extracted["images"]):
        docs.append(Document(
            text=img["description"],
            metadata={
                "content_type": "image_description",
                "page": img["page"],
                "image_path": img["image_path"],
            },
        ))

    return docs


def build_text_index() -> VectorStoreIndex:
    """Construit l'index LlamaIndex sur la collection texte Chroma."""
    with open(EXTRACTED, encoding="utf-8") as f:
        extracted = json.load(f)

    documents = build_documents(extracted)
    print(f"→ {len(documents)} Documents LlamaIndex préparés")

    # Client Chroma persistant
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset propre
    try:
        client.delete_collection(COLLECTION_TEXT)
    except Exception:
        pass
    # ← LE CHANGEMENT : créer avec la MÊME EF que l'agent
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_TEXT)
    chroma_collection = client.create_collection(
        COLLECTION_TEXT,
        embedding_function=embed_fn,
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_TEXT)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print(f"→ Collection '{COLLECTION_TEXT}' : {chroma_collection.count()} vecteurs")
    return index


def build_images_collection() -> None:
    """Encode les images avec CLIP (hors LlamaIndex, comme avant)."""
    from pathlib import Path
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    with open(EXTRACTED, encoding="utf-8") as f:
        extracted = json.load(f)
    images = extracted["images"]

    if not images:
        print("→ Aucune image, CLIP ignoré")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_IMAGES)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION_IMAGES, metadata={"hnsw:space": "cosine"})

    clip = SentenceTransformer(CLIP_MODEL)
    ids, embs, metas = [], [], []
    for j, img in enumerate(images):
        path = Path(img["image_path"])
        if not path.exists():
            continue
        pil = Image.open(path).convert("RGB")
        emb = clip.encode(pil, normalize_embeddings=True).tolist()
        ids.append(f"img_p{img['page']}_i{j}")
        embs.append(emb)
        metas.append({
            "page": img["page"],
            "image_path": img["image_path"],
            "description": img.get("description", ""),
        })
    if ids:
        coll.add(ids=ids, embeddings=embs, metadatas=metas)
    print(f"→ Collection '{COLLECTION_IMAGES}' : {coll.count()} images")


def build_all() -> None:
    print("→ Construction de l'index texte via LlamaIndex…")
    build_text_index()
    print("→ Construction de la collection images via CLIP…")
    build_images_collection()
    print("\n✓ Indexation LlamaIndex terminée.")


if __name__ == "__main__":
    build_all()