import chromadb

from src.embeddings import embed_queries
from src.ingest import COLLECTION, DB_PATH


def buscar(pergunta: str, k: int = 3) -> list[dict]:
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
    res = col.query(query_embeddings=embed_queries([pergunta]), n_results=k)
    return [
        {"id": id_, "texto": doc, "metadados": meta, "distancia": dist}
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
