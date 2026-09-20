from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import hybrid_search

app=FastAPI(title="GeoSearch Research API")
docs=pd.read_csv("artifacts/geophysics_corpus.csv")
bm=BM25Retriever(docs)
dense=DenseRetriever(docs)

class Query(BaseModel):
    query: str
    top_k: int=5

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/search")
def search(q: Query):
    ranked=hybrid_search(bm,dense,q.query,k=q.top_k,alpha=0.6)
    by_id=docs.set_index("doc_id")
    return {"results":[{"doc_id":d,"score":s,"title":by_id.loc[d,"title"],"text":by_id.loc[d,"text"]} for d,s in ranked]}
