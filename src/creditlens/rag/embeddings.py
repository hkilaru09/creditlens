"""Local embeddings via sentence-transformers — no external API or key."""
from __future__ import annotations

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(list(texts), normalize_embeddings=True).tolist()
