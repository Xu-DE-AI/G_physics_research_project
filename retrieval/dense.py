import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from models.encoder import TextEncoder, get_device, MODEL_NAME

class DenseRetriever:
    def __init__(self, docs, checkpoint=None):
        self.docs = docs.reset_index(drop=True)
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME if checkpoint is None else f"{checkpoint}/tokenizer")
        self.model = TextEncoder(MODEL_NAME).to(self.device)
        if checkpoint and os.path.exists(f"{checkpoint}/model.pt"):
            self.model.load_state_dict(torch.load(f"{checkpoint}/model.pt", map_location=self.device))
        self.model.eval()
        self.embeddings = self._encode(self.docs.text.tolist())

    @torch.no_grad()
    def _encode(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            batch = self.tokenizer(texts[i:i+32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            batch = {k:v.to(self.device) for k,v in batch.items()}
            out.append(self.model(**batch).cpu().numpy())
        return np.vstack(out)

    def search(self, query, k=10):
        q = self._encode([query])[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [(self.docs.iloc[i].doc_id, float(scores[i])) for i in order]
