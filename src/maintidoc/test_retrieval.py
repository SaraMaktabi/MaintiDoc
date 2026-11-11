# src/maintidoc/test_retrieval.py
"""
Test rapide de retrieval — sans LLM, sans agent.
Vérifie que Chroma retourne les bons chunks pour quelques questions.
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from maintidoc.config import (
    CHROMA_DIR, COLLECTION_TEXT, COLLECTION_IMAGES,
    EMBED_MODEL_TEXT, CLIP_MODEL,
)


QUESTIONS = [
    "What is the Transformer architecture?",
    "What is multi-head attention?",
    "What is positional encoding?",
    "What is the role of the encoder and decoder?",
    "What dataset was used for translation experiments?",
]


def show(results, label: str, n: int = 3):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    for rank in range(min(n, len(results["ids"][0]))):
        doc_id = results["ids"][0][rank]
        dist = results["distances"][0][rank]
        meta = results["metadatas"][0][rank]
        doc = (results["documents"][0][rank] or "")[:200].replace("\n", " ")
        print(f"\n[{rank+1}] id={doc_id}  distance={dist:.4f}")
        print(f"    content_type={meta.get('content_type')}  page={meta.get('page')}")
        print(f"    {doc}...")


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_TEXT)
    text_coll = client.get_collection(COLLECTION_TEXT, embedding_function=embed_fn)

    print(f"Collection texte : {text_coll.count()} documents\n")

    for q in QUESTIONS:
        r = text_coll.query(query_texts=[q], n_results=3)
        show(r, f"Question : {q}")

    # Test CLIP à part : chercher la figure correspondant à un concept
    print("\n\n" + "#"*70)
    print("# Recherche visuelle CLIP")
    print("#"*70)

    from sentence_transformers import SentenceTransformer
    clip = SentenceTransformer(CLIP_MODEL)
    img_coll = client.get_collection(COLLECTION_IMAGES)

    for q in ["a diagram of a neural network", "attention mechanism schema"]:
        emb = clip.encode(q, normalize_embeddings=True).tolist()
        r = img_coll.query(query_embeddings=[emb], n_results=3)
        print(f"\n{'='*70}\nRequête CLIP : {q}\n{'='*70}")
        for rank in range(min(3, len(r["ids"][0]))):
            print(
                f"  [{rank+1}] id={r['ids'][0][rank]} "
                f"dist={r['distances'][0][rank]:.4f} "
                f"page={r['metadatas'][0][rank].get('page')}"
            )


if __name__ == "__main__":
    main()