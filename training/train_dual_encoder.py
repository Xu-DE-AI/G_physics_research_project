import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.encoder import TextEncoder, get_device, MODEL_NAME

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class PairDataset(Dataset):
    def __init__(self, corpus, queries):
        by_id = corpus.set_index("doc_id")
        self.rows = []
        for q in queries:
            # Keep one positive; the other batch documents become negatives.
            did = q["relevant_doc_ids"][0]
            if did in by_id.index:
                self.rows.append((q["query"], by_id.loc[did, "text"]))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    args = ap.parse_args()

    corpus = pd.read_csv("artifacts/geophysics_corpus.csv")
    queries = json.load(open("artifacts/queries.json", encoding="utf-8"))
    ds = PairDataset(corpus, queries)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TextEncoder(MODEL_NAME).to(get_device())
    device = get_device()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for qs, docs in loader:
            q = tokenizer(list(qs), padding=True, truncation=True, max_length=128, return_tensors="pt")
            d = tokenizer(list(docs), padding=True, truncation=True, max_length=256, return_tensors="pt")
            q = {k: v.to(device) for k, v in q.items()}
            d = {k: v.to(device) for k, v in d.items()}
            q_emb = model(**q)
            d_emb = model(**d)
            logits = q_emb @ d_emb.T / args.temperature
            labels = torch.arange(len(qs), device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch={epoch+1} loss={np.mean(losses):.4f}")

    os.makedirs("artifacts/dual_encoder", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/dual_encoder/model.pt")
    tokenizer.save_pretrained("artifacts/dual_encoder/tokenizer")
    with open("artifacts/dual_encoder/config.json", "w") as f:
        json.dump({"base_model": MODEL_NAME, "projection_dim": 256}, f)
    print("Saved artifacts/dual_encoder")

if __name__ == "__main__": main()
