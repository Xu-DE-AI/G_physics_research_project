import re
import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[A-Za-zÀ-ÿ0-9-]+", text.lower())


class BM25Retriever:
    def __init__(self, docs: pd.DataFrame):
        self.docs = docs.reset_index(drop=True)
        self.tokens = [tokenize(x) for x in self.docs.text]
        self.model = BM25Okapi(self.tokens)

    def search(self, query, k=10):
        scores = self.model.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(self.docs.iloc[i].doc_id, float(scores[i])) for i in order]
