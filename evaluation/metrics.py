import math

def recall_at_k(ranked, relevant, k):
    return float(any(d in set(relevant) for d, _ in ranked[:k]))

def mrr(ranked, relevant):
    rel = set(relevant)
    for i, (d, _) in enumerate(ranked, 1):
        if d in rel: return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, relevant, k):
    rel = set(relevant)
    dcg = sum((1.0 / math.log2(i+2)) for i,(d,_) in enumerate(ranked[:k]) if d in rel)
    ideal_hits = min(k, len(rel))
    idcg = sum(1.0 / math.log2(i+2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
