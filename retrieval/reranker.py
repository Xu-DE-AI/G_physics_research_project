"""Optional second-stage cross-encoder reranker."""
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, k=10):
        # candidates: [(doc_id, document_text), ...]
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = [(doc_id, float(score)) for (doc_id, _), score in zip(candidates, scores)]
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:k]
