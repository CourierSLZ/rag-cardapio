from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.generator import responder
from src.retriever import buscar

app = FastAPI(title="RAG Cardápio")


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)


@app.post("/perguntar")
def perguntar(p: Pergunta):
    trechos = buscar(p.pergunta, p.k)
    resposta = responder(p.pergunta, [t["texto"] for t in trechos])
    return {"resposta": resposta, "fontes": [{"id": t["id"], "distancia": t["distancia"]} for t in trechos]}


@app.get("/saude")
def saude():
    return {"status": "ok"}


# Front-end simples (HTML + JS) servido na raiz. Mantenha por último.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
