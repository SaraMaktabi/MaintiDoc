# src/maintidoc/agent.py
"""
Agent LangChain v1 pour MaintiDoc.
Qwen2.5:7b via Ollama choisit parmi les outils de recherche.
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from maintidoc.config import (
    CHROMA_DIR, COLLECTION_TEXT, COLLECTION_IMAGES,
    EMBED_MODEL_TEXT, CLIP_MODEL, LLM_MODEL, OLLAMA_URL,
)
from maintidoc.rag import format_context


# ---------- Client Chroma partagé ----------
_client = None
_text_coll = None
_img_coll = None


def _get_text_coll():
    global _client, _text_coll
    if _text_coll is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_TEXT)
        _text_coll = _client.get_collection(
            COLLECTION_TEXT, embedding_function=embed_fn
        )
    return _text_coll


def _get_img_coll():
    global _client, _img_coll
    if _img_coll is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _img_coll = _client.get_collection(COLLECTION_IMAGES)
    return _img_coll


# ---------- Fonctions de recherche ----------

def _query_by_type(question: str, content_type: str, n: int) -> str:
    coll = _get_text_coll()
    try:
        res = coll.query(
            query_texts=[question],
            n_results=n,
            where={"content_type": content_type},
        )
    except Exception as e:
        return f"(erreur de recherche : {e})"

    if not res["ids"] or not res["ids"][0]:
        return f"(aucun passage de type '{content_type}' trouvé)"

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],
            "metadata": res["metadatas"][0][i] or {},
            "text": res["documents"][0][i] or "",
            "content_type": content_type,
        })
    return format_context(hits)


def _search_clip_impl(question: str, n: int = 3) -> str:
    from sentence_transformers import SentenceTransformer
    clip = SentenceTransformer(CLIP_MODEL)
    emb = clip.encode(question, normalize_embeddings=True).tolist()
    coll = _get_img_coll()
    try:
        res = coll.query(query_embeddings=[emb], n_results=n)
    except Exception as e:
        return f"(erreur de recherche CLIP : {e})"

    if not res["ids"] or not res["ids"][0]:
        return "(aucune image trouvée)"

    lines = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        lines.append(
            f"[Image {i+1} | page {meta.get('page', '?')} | "
            f"dist={res['distances'][0][i]:.3f}]\n"
            f"chemin: {meta.get('image_path', '?')}\n"
            f"description: {meta.get('description', '')[:300]}"
        )
    return "\n\n---\n\n".join(lines)


# ---------- Tools LangChain v1 (décorateur @tool) ----------

@tool
def search_text(query: str) -> str:
    """Recherche dans les passages de texte du document technique.
    À utiliser pour les questions sur les concepts, définitions, procédures, explications.
    C'est l'outil principal."""
    return _query_by_type(query, "text", n=6)


@tool
def search_tables(query: str) -> str:
    """Recherche dans les tableaux du document.
    À utiliser pour les questions sur des valeurs numériques, comparaisons,
    spécifications, métriques, hyperparamètres."""
    return _query_by_type(query, "table", n=3)


@tool
def search_image_descriptions(query: str) -> str:
    """Recherche dans les descriptions textuelles des images et figures.
    À utiliser pour les questions sur ce que montre une figure, un schéma, un diagramme."""
    return _query_by_type(query, "image_description", n=3)


@tool
def search_clip(query: str) -> str:
    """Recherche visuelle par similarité CLIP dans les images du document.
    À utiliser pour trouver une image à partir d'une description visuelle
    (ex: 'schéma d'un réseau de neurones'). Renvoie les chemins des images les plus proches."""
    return _search_clip_impl(query, n=3)


TOOLS = [search_text, search_tables, search_image_descriptions, search_clip]


# ---------- Prompt système ----------

SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance.
Tu réponds UNIQUEMENT à partir des informations retournées par les outils.
Tu ne pilotes aucun équipement et tu ne remplaces pas un technicien habilité.

Règles :
1. Choisis le ou les outils les plus adaptés (texte pour les concepts, tableaux pour les valeurs,
   descriptions d'images pour les figures).
2. Si un outil ne retourne pas d'information pertinente, ESSAIE UN AUTRE OUTIL
   avant de conclure. Par exemple, si search_tables ne trouve rien, essaie search_text.
3. Pour les questions complexes ou composées (comparaisons, explications multiples),
   appelle PLUSIEURS outils : par exemple search_tables pour les chiffres
   ET search_text pour les explications.
4. Tu peux appeler plusieurs outils si la question est complexe.
5. Cite tes sources sous la forme [Source N] après chaque affirmation importante.
6. Si une figure est pertinente, mentionne simplement « voir figure page X ».
   N'insère JAMAIS de markdown d'image ![...](...).
7. Si AUCUN outil ne retourne d'information pertinente après avoir essayé,
   dis clairement : « Je ne trouve pas cette information dans le document. »
8. Réponds en français, de façon concise et technique."""


# ---------- Construction de l'agent ----------

def build_agent():
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
        num_ctx=8192,
    )
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# ---------- Point d'entrée ----------

def answer(question: str) -> dict:
    agent = build_agent()
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        messages = result.get("messages", [])
        final = messages[-1].content if messages else ""
        # Extraire les appels d'outils pour debug
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        return {"question": question, "answer": final, "tool_calls": tool_calls, "error": None}
    except Exception as e:
        return {"question": question, "answer": f"[erreur agent] {e}", "tool_calls": [], "error": str(e)}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many attention heads does the base model use?"
    print(f"\n>>> Question : {q}\n")
    result = answer(q)
    print(result["answer"])
    if result["tool_calls"]:
        print("\n=== Outils appelés ===")
        for tc in result["tool_calls"]:
            print(f"- {tc['name']}({tc['args']})")