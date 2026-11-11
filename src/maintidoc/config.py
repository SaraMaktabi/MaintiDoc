# src/maintidoc/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
EXTRACTED  = DATA_DIR / "extracted.json"
CHROMA_DIR = ROOT / "chroma_db"

# Création des dossiers au chargement (évite les erreurs plus tard)
for _d in (DATA_DIR, RAW_DIR, IMAGES_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PDF_PATH = RAW_DIR / "attention_is_all_you_need.pdf"

# Collections Chroma
COLLECTION_TEXT   = "maintidoc_text"     # texte + tableaux + descriptions d'images
COLLECTION_IMAGES = "maintidoc_images"   # embeddings CLIP

# Modèles
EMBED_MODEL_TEXT = "BAAI/bge-small-en-v1.5"          # 384d
CLIP_MODEL       = "openai/clip-vit-base-patch32"    # 512d
LLM_MODEL        = "qwen2.5:3b-instruct"
OLLAMA_URL       = "http://localhost:11434"