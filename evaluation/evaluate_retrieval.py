import json, time
import pandas as pd
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import hybrid_search
from evaluation.metrics import recall_at_k, mrr, ndcg_at_k


def evaluate(name, search_fn, queries):
    rows=[]
    for q in queries:
        t=time.perf_counter(); ranked=search_fn(q["query"], 10); ms=(time.perf_counter()-t)*1000
        rows.append({"recall@5":recall_at_k(ranked,q["relevant_doc_ids"],5), "recall@10":recall_at_k(ranked,q["relevant_doc_ids"],10), "mrr":mrr(ranked,q["relevant_doc_ids"]), "ndcg@10":ndcg_at_k(ranked,q["relevant_doc_ids"],10), "latency_ms":ms})
    df=pd.DataFrame(rows)
    out=df.mean(numeric_only=True).to_dict(); out["system"]=name
    return out


def main():
    docs=pd.read_csv("artifacts/geophysics_corpus.csv")
    queries=json.load(open("artifacts/queries.json", encoding="utf-8"))
    bm=BM25Retriever(docs)
    dense=DenseRetriever(docs)
    systems=[
        ("BM25", lambda q,k: bm.search(q,k)),
        ("Dense", lambda q,k: dense.search(q,k)),
        ("Hybrid", lambda q,k: hybrid_search(bm,dense,q,k,alpha=0.6)),
    ]
    results=[evaluate(n,f,queries) for n,f in systems]
    out=pd.DataFrame(results)
    print(out.to_string(index=False))
    out.to_csv("artifacts/retrieval_results.csv",index=False)

if __name__ == "__main__": main()
