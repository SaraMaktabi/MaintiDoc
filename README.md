# MaintiDoc

## Nouveauté : photo → OCR → procédure

Le technicien peut charger une photo de pièce ou de plaque signalétique. MaintiDoc extrait le texte visible avec Tesseract, compare visuellement la photo aux images du manuel avec CLIP, puis utilise le texte OCR pour rechercher la référence ou la procédure dans l'index documentaire.

Exemple CLI :

```powershell
uv run python -m maintidoc.main ask-photo "Quelle est cette référence et quelle procédure appliquer ?" data/photo.jpg
```

Dans Streamlit, charger le PDF, extraire et indexer, puis ajouter la photo dans le panneau « Contexte visuel ». Le texte OCR peut être corrigé avant l'envoi de la question.
