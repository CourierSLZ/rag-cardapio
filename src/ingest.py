"""Indexa o cardápio no Chroma. Rode uma vez (ou quando o cardápio mudar)."""
import json
from pathlib import Path

import chromadb

from src.embeddings import embed_passages

DB_PATH = "chroma_db"
COLLECTION = "cardapio"


def item_para_texto(item: dict) -> str:
    return (
        f"{item['nome']} ({item['categoria']}). {item['descricao']} "
        f"Preço: R$ {item['preco']:.2f}. {item['restricoes']}"
    )


def main(caminho: str = "data/cardapio.json") -> None:
    itens = json.loads(Path(caminho).read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    textos = [item_para_texto(i) for i in itens]
    col.add(
        ids=[i["id"] for i in itens],
        documents=textos,
        embeddings=embed_passages(textos),
        metadatas=[
            {"nome": i["nome"], "categoria": i["categoria"], "preco": float(i["preco"])}
            for i in itens
        ],
    )
    print(f"{len(itens)} itens indexados em {DB_PATH}/")


if __name__ == "__main__":
    main()
