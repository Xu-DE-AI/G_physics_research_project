import argparse, json
import pandas as pd
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import hybrid_search


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("query", nargs="+"); args=ap.parse_args()
    query=" ".join(args.query)
    docs=pd.read_csv("artifacts/geophysics_corpus.csv")
    bm=BM25Retriever(docs); dense=DenseRetriever(docs)
    ranked=hybrid_search(bm,dense,query,k=5,alpha=0.6)
    by_id=docs.set_index("doc_id")
    print("\nRetrieved evidence:")
    for i,(doc,score) in enumerate(ranked,1):
        print(f"[{i}] {doc} score={score:.3f}\n{by_id.loc[doc,'text']}\n")
    print("RAG prompt to use with a local/hosted LLM:\n")
    context="\n".join([f"[{d}] {by_id.loc[d,'text']}" for d,_ in ranked[:5]])
    print(f"Answer the question using ONLY the evidence below. Cite document IDs.\n\nQuestion: {query}\n\nEvidence:\n{context}")

if __name__ == "__main__": main()
