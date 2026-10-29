"""Métricas de recuperação (retrieval)."""


def hit_at_k(esperados: set[str], recuperados: list[str], k: int) -> float:
    """1.0 se pelo menos um item esperado aparece nos k primeiros resultados."""
    return float(any(r in esperados for r in recuperados[:k]))


def reciprocal_rank(esperados: set[str], recuperados: list[str]) -> float:
    for pos, r in enumerate(recuperados, start=1):
        if r in esperados:
            return 1.0 / pos
    return 0.0
