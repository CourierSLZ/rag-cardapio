# RAG: atendente de cardápio

Chatbot que responde perguntas sobre um cardápio usando **RAG**:
embeddings multilíngues + Chroma para recuperar os itens relevantes e um LLM para
redigir a resposta usando apenas esse contexto. Inclui **avaliação da recuperação**
(Hit@k e MRR) e um front-end em HTML/JS.

```
pergunta → embedding → Chroma (top-k) → LLM com contexto → resposta + fontes
```

## Como rodar
```bash
pip install -r requirements.txt
cp .env.example .env && export ANTHROPIC_API_KEY=...   # chave do LLM
python -m src.ingest
uvicorn src.api:app --reload      # abre http://localhost:8000
python -m eval.eval_retrieval     # métricas de recuperação
pytest -q
```
Docker: `docker build -t rag-cardapio . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... rag-cardapio`

## Avaliação
| Configuração | Hit@3 | MRR |
|---|---|---|
| e5-small, 1 chunk por item | _preencher_ | _preencher_ |
| (sua variação: outro modelo / texto do chunk) | _preencher_ | _preencher_ |

Compare também respostas **com** e **sem** contexto (`responder` vs `responder_sem_contexto`)
em perguntas sobre preço e restrições, onde o LLM puro tende a inventar.

## Limitações
- Cardápio pequeno (10 itens): o ganho do RAG aparece mais com bases grandes.
- Conjunto golden pequeno; aumente-o antes de tirar conclusões.

## Próximos passos
- Busca híbrida (BM25 + vetorial) e reranking.
- Avaliar fidelidade da resposta (LLM-as-judge) além da recuperação.
