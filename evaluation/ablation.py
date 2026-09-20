"""Research notebook replacement: print a checklist for controlled ablations."""
EXPERIMENTS=[
    ("A", "BM25", "lexical baseline"),
    ("B", "Dense frozen", "pretrained representation"),
    ("C", "Dense contrastive", "learned representation"),
    ("D", "Hybrid", "sparse+dense fusion"),
    ("E", "Hybrid + reranker", "second-stage ranking"),
    ("F", "Hybrid + hard negatives", "training-data optimization"),
    ("G", "Text + waveform", "multimodal representation fusion"),
    ("H", "English + German", "cross-lingual evaluation"),
]

def main():
    print("Run each experiment with the SAME fixed query split. Record Recall@5, MRR, nDCG@10, p50/p95 latency.")
    for x in EXPERIMENTS: print(" | ".join(x))

if __name__=="__main__": main()
