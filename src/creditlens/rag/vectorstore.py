"""Thin wrapper around a local, on-disk Chroma collection."""
from __future__ import annotations


class FilingVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "filings"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], n_results: int = 5, where: dict | None = None) -> dict:
        return self._collection.query(
            query_embeddings=[embedding], n_results=n_results, where=where
        )

    def count(self) -> int:
        return self._collection.count()
