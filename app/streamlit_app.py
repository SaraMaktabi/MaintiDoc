# app/streamlit_app.py
"""Parcours terrain : PDF → extraction → indexation → photo → OCR → question sourcée."""
import sys
from pathlib import Path

import streamlit as st

# Permettre l'import du package src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintidoc.config import RAW_DIR, PDF_PATH, IMAGES_DIR, EXTRACTED, CHROMA_DIR


# ---------- État de session ----------

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "uploaded_photo_path" not in st.session_state:
        st.session_state.uploaded_photo_path = None


# ---------- Actions d'indexation ----------

def run_extraction(pdf_file) -> bool:
    """Sauvegarde le PDF uploadé puis lance extract.py."""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        target = RAW_DIR / "uploaded.pdf"
        target.write_bytes(pdf_file.getbuffer())

        # Copier aussi sous le nom attendu par config.py
        PDF_PATH.write_bytes(pdf_file.getbuffer())

        # Nettoyer les anciennes images
        if IMAGES_DIR.exists():
            for f in IMAGES_DIR.glob("*.png"):
                f.unlink()

        from maintidoc.extract import run_extraction as do_extract
        result = do_extract()
        st.success(
            f"Extraction OK — {len(result['pages'])} pages, "
            f"{len(result['tables'])} tableaux, {len(result['images'])} images."
        )
        return True
    except Exception as e:
        st.error(f"Erreur d'extraction : {e}")
        return False


def run_indexing() -> bool:
    """Lance l'indexation LlamaIndex (texte + CLIP)."""
    try:
        # Reset Chroma pour éviter les conflits d'embedding
        import shutil
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        from maintidoc.index_llama import build_all
        build_all()
        st.session_state.indexed = True
        st.success("Indexation terminée.")
        return True
    except Exception as e:
        st.error(f"Erreur d'indexation : {e}")
        return False


# ---------- Affichage d'une réponse ----------

def render_sources(tool_calls, question: str):
    """Affiche les outils appelés et, si possible, les images pertinentes."""
    if not tool_calls:
        return
    with st.expander("🔧 Outils appelés par l'agent", expanded=False):
        for tc in tool_calls:
            st.markdown(f"- **`{tc['name']}`** — `{tc['args']}`")

    # Si un outil CLIP a été utilisé, afficher les images
    if any(tc["name"] == "search_clip" for tc in tool_calls):
        with st.expander("🖼️ Images pertinentes", expanded=True):
            _show_clip_images(question)


def _show_clip_images(question: str, n: int = 3):
    """Interroge la collection CLIP et affiche les images retournées."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from maintidoc.config import COLLECTION_IMAGES, CLIP_MODEL

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        coll = client.get_collection(COLLECTION_IMAGES)
        clip = SentenceTransformer(CLIP_MODEL)
        emb = clip.encode(question, normalize_embeddings=True).tolist()
        res = coll.query(query_embeddings=[emb], n_results=n)

        cols = st.columns(len(res["ids"][0]))
        for i, col in enumerate(cols):
            meta = res["metadatas"][0][i]
            path = Path(meta.get("image_path", ""))
            with col:
                if path.exists():
                    st.image(str(path), caption=f"page {meta.get('page')}", use_container_width=True)
                else:
                    st.caption(f"(image absente : {path.name})")
    except Exception as e:
        st.caption(f"(impossible d'afficher les images : {e})")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            render_sources(msg["tool_calls"], msg.get("question", ""))


def render_visual_context(photo):
    """Affiche la photo terrain et le texte OCR que le technicien peut corriger."""
    st.subheader("📷 Contexte visuel")
    if photo is None:
        st.info("Ajoutez une photo d'une pièce ou d'une plaque signalétique.")
        return
    photo_path = RAW_DIR / "user_photo.png"
    photo_path.write_bytes(photo.getbuffer())
    st.session_state.uploaded_photo_path = str(photo_path)
    st.image(photo, caption="Photo utilisateur", use_container_width=True)
    try:
        from maintidoc.agent import _ocr_photo
        if not st.session_state.ocr_text:
            detected = _ocr_photo(str(photo_path))
            st.session_state.ocr_text = "" if detected.startswith("(OCR indisponible") else detected
    except Exception as exc:
        st.warning(f"OCR indisponible : {exc}")
    st.session_state.ocr_text = st.text_area(
        "Texte OCR (modifiable)", value=st.session_state.ocr_text,
        key="ocr_editor", height=120,
    )


# ---------- Page principale ----------

def main():
    st.set_page_config(page_title="MaintiDoc", page_icon="🔧", layout="wide")
    init_state()
    st.markdown("""<style>
    .main-title {background:linear-gradient(90deg,#125b8a,#52616b);color:white;padding:1rem 1.4rem;border-radius:12px;margin-bottom:1rem}
    .status-ready {color:#137333;font-weight:700}.status-warn {color:#b06000;font-weight:700}
    [data-testid="stChatMessage"] {border-radius:12px}
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="main-title"><h1>🔧 MaintiDoc</h1><p>Assistant documentaire de maintenance — RAG multimodal agentique</p></div>', unsafe_allow_html=True)
    st.caption(
        "RAG multimodal agentique — LlamaIndex + ChromaDB + CLIP + Qwen via Ollama. "
        "Ne pilote aucun équipement et ne remplace pas un technicien habilité."
    )

    # ----- Sidebar : upload + indexation -----
    with st.sidebar:
        st.header("📄 Document")
        uploaded = st.file_uploader("Charger un PDF", type=["pdf"])

        col1, col2 = st.columns(2)
        with col1:
            extract_btn = st.button("1. Extraire", disabled=uploaded is None, use_container_width=True)
        with col2:
            index_btn = st.button("2. Indexer", disabled=not EXTRACTED.exists(), use_container_width=True)

        st.divider()
        st.markdown("**État**")
        st.write(f"- PDF chargé : {'✅' if PDF_PATH.exists() else '❌'}")
        st.write(f"- Extraction : {'✅' if EXTRACTED.exists() else '❌'}")
        st.write(f"- Index prêt : {'✅' if st.session_state.indexed else '❌'}")

        if st.button("🗑️ Réinitialiser le chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.ocr_text = ""
            st.session_state.uploaded_photo_path = None
            st.rerun()

    # ----- Actions -----
    if extract_btn and uploaded:
        with st.spinner("Extraction en cours…"):
            run_extraction(uploaded)

    if index_btn:
        with st.spinner("Indexation LlamaIndex en cours (peut prendre 1–3 min)…"):
            run_indexing()

    if not st.session_state.indexed:
        st.info("Chargez un PDF, cliquez **1. Extraire** puis **2. Indexer** pour commencer.")
        return

    photo = st.file_uploader("📷 Photo de la pièce ou du panneau", type=["png", "jpg", "jpeg"], key="photo_uploader")
    left, right = st.columns([3, 2])
    with right:
        render_visual_context(photo)
    with left:
        for msg in st.session_state.messages:
            render_message(msg)

    if question := st.chat_input("Posez votre question sur le document…"):
        photo_path = st.session_state.uploaded_photo_path
        if st.session_state.ocr_text:
            question += f"\n\n[Texte OCR corrigé par le technicien : {st.session_state.ocr_text}]"
        st.session_state.messages.append({"role": "user", "content": question})
        render_message({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit…"):
                try:
                    from maintidoc.agent import answer as agent_answer
                    result = agent_answer(question, photo_path=photo_path)
                    answer_text = result["answer"]
                    tool_calls = result.get("tool_calls", [])
                except Exception as e:
                    st.error(f"Erreur agent : {e}")
                    # Fallback : RAG simple
                    try:
                        from maintidoc.rag import answer as rag_answer
                        r = rag_answer(question)
                        answer_text = r["answer"]
                        tool_calls = []
                    except Exception as e2:
                        answer_text = f"Erreur : {e2}"
                        tool_calls = []

            st.markdown(answer_text)
            if tool_calls:
                render_sources(tool_calls, question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "tool_calls": tool_calls,
            "question": question,
            "ocr_text": result.get("ocr_text", st.session_state.ocr_text) if 'result' in locals() else st.session_state.ocr_text,
        })


if __name__ == "__main__":
    main()
