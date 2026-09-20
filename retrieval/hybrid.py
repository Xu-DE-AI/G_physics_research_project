import numpy as np


def normalize_scores(items):
    if not items: return {}
    vals = np.array([x[1] for x in items], dtype=float)
    lo, hi = vals.min(), vals.max()
    denom = hi - lo if hi > lo else 1.0
    return {doc: float((score-lo)/denom) for doc, score in items}


def hybrid_search(bm25, dense, query, k=10, alpha=0.5):
    b = normalize_scores(bm25.search(query, max(k, 20)))
    d = normalize_scores(dense.search(query, max(k, 20)))
    ids = set(b) | set(d)
    scores = {doc: alpha*d.get(doc, 0.0) + (1-alpha)*b.get(doc, 0.0) for doc in ids}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
