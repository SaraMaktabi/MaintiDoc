# src/maintidoc/agent.py
"""
Agent LangChain v1 pour MaintiDoc.
Qwen2.5:7b via Ollama choisit parmi les outils de recherche.
"""
import chromadb
import os
from pathlib import Path
import pytesseract
from PIL import Image
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


def _ocr_photo(photo_path: str) -> str:
    """Extrait le texte visible d'une photo sans interrompre l'agent."""
    try:
        candidates = [os.getenv("TESSERACT_CMD"), r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        return pytesseract.image_to_string(
            Image.open(photo_path).convert("RGB"), lang="fra+eng", config="--psm 6"
        ).strip()
    except Exception as exc:
        return f"(OCR indisponible : {exc})"


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


def _search_clip_photo_impl(photo_path: str, n: int = 3) -> str:
    """Compare une photo utilisateur aux images du corpus avec CLIP."""
    from pathlib import Path
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    path = Path(photo_path)
    if not path.exists():
        return "(photo utilisateur introuvable)"
    clip = SentenceTransformer(CLIP_MODEL)
    emb = clip.encode(Image.open(path).convert("RGB"), normalize_embeddings=True).tolist()
    try:
        res = _get_img_coll().query(query_embeddings=[emb], n_results=n)
    except Exception as e:
        return f"(erreur de recherche de la photo : {e})"
    if not res["ids"] or not res["ids"][0]:
        return "(aucune image comparable trouvée)"
    lines = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        lines.append(f"[Correspondance photo {i + 1} | page {meta.get('page', '?')} | dist={res['distances'][0][i]:.3f}]\nchemin: {meta.get('image_path', '?')}\ndescription: {meta.get('description', '')[:500]}")
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


@tool
def search_clip_photo(photo_path: str) -> str:
    """Compare une photo fournie par le technicien avec les images du document.
    Le chemin de la photo est indiqué dans le message utilisateur entre crochets.
    À utiliser quand l'utilisateur mentionne une photo ou une image à analyser."""
    return _search_clip_photo_impl(photo_path, n=3)


TOOLS = [search_text, search_tables, search_image_descriptions, search_clip, search_clip_photo]


# ---------- Prompt système ----------

SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance.
Si une photo est fournie, utilise search_clip_photo pour l'identifier visuellement, puis search_text
avec le texte OCR pour retrouver la référence ou la procédure associée et cite les pages.
Tu réponds UNIQUEMENT à partir des informations retournées par les outils.
Tu ne pilotes aucun équipement et tu ne remplaces pas un technicien habilité.

RÈGLE ABSOLUE : Si les outils ne retournent pas l'information demandée,
tu réponds UNIQUEMENT « Je ne trouve pas cette information dans le document. »
Tu N'AJOUTES RIEN d'autre. Tu n'utilises PAS tes connaissances générales.
Tu n'inventes AUCUNE procédure, AUCUNE valeur, AUCUNE recette.
Même si tu connais la réponse par ailleurs, tu ne la donnes PAS.

Autres règles :
1. Choisis le ou les outils les plus adaptés (texte pour les concepts, tableaux pour les valeurs,
   descriptions d'images pour les figures).
2. Si un outil ne retourne rien de pertinent, essaie un autre outil avant de conclure.
3. Pour les questions composées, appelle plusieurs outils.
4. Cite tes sources [Source N] après chaque affirmation.
5. Pour une figure, mentionne « voir figure page X ». Pas de markdown d'image.
6. Réponds en français, de façon concise et technique."""


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

def answer(question: str, photo_path: str | None = None) -> dict:
    agent = build_agent()
    ocr_text = ""
    try:
        if photo_path:
            ocr_text = _ocr_photo(photo_path)
            user_content = (
                f"{question}\n\n"
                f"[Texte OCR détecté : {ocr_text or '(aucun texte détecté)'}]\n"
                f"[Une photo utilisateur est disponible au chemin : {photo_path}]\n"
                f"[Utilise OBLIGATOIREMENT l'outil search_clip_photo avec ce chemin "
                f"pour la comparer aux images du document avant de répondre.]"
            )
        else:
            user_content = question

        result = agent.invoke({
            "messages": [{"role": "user", "content": user_content}]
        })

        messages = result.get("messages", [])
        final = messages[-1].content if messages else ""

        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})

        return {"question": question, "answer": final, "tool_calls": tool_calls,
                "ocr_text": ocr_text, "error": None}
    except Exception as e:
        return {"question": question, "answer": f"[erreur agent] {e}", "tool_calls": [],
                "ocr_text": ocr_text, "error": str(e)}


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
