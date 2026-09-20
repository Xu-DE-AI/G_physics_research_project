from evaluation.metrics import recall_at_k, mrr, ndcg_at_k

def test_metrics():
    ranked=[("a",.9),("b",.8),("c",.7)]
    rel=["b"]
    assert recall_at_k(ranked,rel,2)==1
    assert mrr(ranked,rel)==0.5
    assert ndcg_at_k(ranked,rel,3)>0
