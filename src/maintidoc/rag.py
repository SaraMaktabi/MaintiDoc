# src/maintidoc/rag.py
"""
RAG simple : question -> Chroma (par type) -> prompt -> Qwen via Ollama.
Pas d'agent, pas de LangChain. Sert de baseline avant l'agentique.
"""
import json
import urllib.request

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from maintidoc.config import (
    CHROMA_DIR, COLLECTION_TEXT,
    EMBED_MODEL_TEXT, LLM_MODEL, OLLAMA_URL,
)


# ---------- Retrieval ----------

def _query_by_type(coll, question: str, content_type: str, n: int) -> list[dict]:
    """Interroge Chroma en filtrant sur content_type. Retourne une liste de hits."""
    try:
        res = coll.query(
            query_texts=[question],
            n_results=n,
            where={"content_type": content_type},
        )
    except Exception as e:
        print(f"[warn] query {content_type}: {e}")
        return []

    hits = []
    if not res["ids"] or not res["ids"][0]:
        return hits
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],
            "metadata": res["metadatas"][0][i] or {},
            "text": res["documents"][0][i] or "",
            "content_type": content_type,
        })
    return hits


def retrieve(question: str, n_text: int = 6, n_table: int = 2, n_img: int = 2) -> list[dict]:
    """Récupère les passages pertinents dans les 3 types de contenu."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_TEXT)
    coll = client.get_collection(COLLECTION_TEXT, embedding_function=embed_fn)

    hits = []
    hits += _query_by_type(coll, question, "text", n_text)
    hits += _query_by_type(coll, question, "table", n_table)
    hits += _query_by_type(coll, question, "image_description", n_img)
    return hits


# ---------- Construction du prompt ----------

def format_context(hits: list[dict]) -> str:
    """Formate les hits en un contexte lisible avec identifiants de source."""
    if not hits:
        return "(aucun passage trouvé)"

    blocks = []
    for i, h in enumerate(hits, start=1):
        page = h["metadata"].get("page", "?")
        ctype = h["content_type"]
        blocks.append(
            f"[Source {i} | {ctype} | page {page} | dist={h['distance']:.3f}]\n"
            f"{h['text'][:900]}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance.
Tu réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
Si le contexte ne contient pas l'information, tu dis clairement « Je ne trouve pas cette information dans le document. »
Tu cites tes sources sous la forme [Source N] après chaque affirmation importante.
Tu réponds en français, de façon concise et technique.
Tu ne pilotes aucun équipement et tu ne remplaces pas un technicien habilité.
"""


def build_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

### Contexte
{context}

### Question
{question}

### Réponse
"""


# ---------- Appel Ollama ----------

def call_ollama(prompt: str, model: str = LLM_MODEL, url: str = OLLAMA_URL,
                temperature: float = 0.1) -> str:
    """Appel direct à l'API Ollama /api/generate (pas de dépendance au SDK)."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 4096},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


# ---------- Pipeline complet ----------

def answer(question: str) -> dict:
    hits = retrieve(question)
    context = format_context(hits)
    prompt = build_prompt(question, context)
    response = call_ollama(prompt)
    return {
        "question": question,
        "answer": response,
        "sources": hits,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the Transformer architecture?"
    print(f"\n>>> Question : {q}\n")
    result = answer(q)
    print(result["answer"])
    print("\n--- Sources ---")
    for i, h in enumerate(result["sources"], start=1):
        print(
            f"[{i}] {h['content_type']} page {h['metadata'].get('page')} "
            f"(dist={h['distance']:.3f}, id={h['id']})"
        )