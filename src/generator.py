"""Geração de resposta com um LLM (Anthropic). Troque o cliente se preferir outro provedor."""
import os

SYSTEM_COM_CONTEXTO = (
    "Você é o atendente virtual de um restaurante. Responda em português, de forma curta, "
    "usando SOMENTE as informações do cardápio fornecido. Se a informação não estiver no "
    "cardápio, diga que não sabe. Nunca invente itens, preços ou ingredientes."
)
SYSTEM_SEM_CONTEXTO = "Você é o atendente virtual de um restaurante. Responda em português, de forma curta."


def _chamar(system: str, user: str) -> str:
    import anthropic  # import tardio: os testes não precisam do SDK nem da chave

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def responder(pergunta: str, contextos: list[str]) -> str:
    contexto = "\n".join(f"- {c}" for c in contextos)
    return _chamar(SYSTEM_COM_CONTEXTO, f"Cardápio relevante:\n{contexto}\n\nPergunta: {pergunta}")


def responder_sem_contexto(pergunta: str) -> str:
    """Usado na avaliação para comparar com e sem RAG."""
    return _chamar(SYSTEM_SEM_CONTEXTO, pergunta)
