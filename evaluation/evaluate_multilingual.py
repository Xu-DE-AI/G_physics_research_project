import json
import pandas as pd
from retrieval.multilingual import MultilingualRetriever
from evaluation.metrics import recall_at_k, mrr, ndcg_at_k

def main():
    docs=pd.read_csv('artifacts/geophysics_corpus.csv')
    queries=[q for q in json.load(open('artifacts/queries.json',encoding='utf-8')) if q.get('language')=='de']
    r=MultilingualRetriever(docs)
    rows=[]
    for q in queries:
        ranked=r.search(q['query'],10)
        rows.append([recall_at_k(ranked,q['relevant_doc_ids'],5),mrr(ranked,q['relevant_doc_ids']),ndcg_at_k(ranked,q['relevant_doc_ids'],10)])
    print(pd.DataFrame(rows,columns=['recall@5','mrr','ndcg@10']).mean())

if __name__=='__main__': main()
