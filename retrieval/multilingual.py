"""Cross-lingual dense retrieval baseline.

Unlike the English DistilBERT research encoder, this baseline uses a multilingual
Sentence Transformer. It is useful for testing whether an English-trained index
can retrieve German evidence and for motivating multilingual contrastive learning.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

class MultilingualRetriever:
    def __init__(self, docs, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.docs=docs.reset_index(drop=True)
        self.model=SentenceTransformer(model_name)
        self.embeddings=self.model.encode(self.docs.text.tolist(), normalize_embeddings=True, show_progress_bar=True)
    def search(self, query, k=10):
        q=self.model.encode([query], normalize_embeddings=True)[0]
        scores=self.embeddings @ q
        order=np.argsort(-scores)[:k]
        return [(self.docs.iloc[i].doc_id,float(scores[i])) for i in order]
