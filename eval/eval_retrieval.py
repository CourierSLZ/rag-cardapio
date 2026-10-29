"""Avalia a recuperação: Hit@k e MRR sobre um conjunto 'golden'.

Uso:  python -m eval.eval_retrieval
Depois de ajustar chunking/embeddings, rode de novo e compare os números no README.
"""
import json
from pathlib import Path

from src.metrics import hit_at_k, reciprocal_rank
from src.retriever import buscar

K = 3


def main() -> None:
    casos = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))
    hits, rrs = [], []
    for c in casos:
        ids = [r["id"] for r in buscar(c["pergunta"], k=10)]
        esperados = set(c["esperados"])
        hits.append(hit_at_k(esperados, ids, K))
        rrs.append(reciprocal_rank(esperados, ids))
        marca = "OK " if hits[-1] else "ERR"
        print(f"[{marca}] {c['pergunta']}  ->  {ids[:K]}")
    print(f"\nHit@{K}: {sum(hits)/len(hits):.2f} | MRR: {sum(rrs)/len(rrs):.2f} | n={len(casos)}")


if __name__ == "__main__":
    main()
