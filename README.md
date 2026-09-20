# GeoSearch Research Lab

**geophysics / induced-seismicity** The role emphasizes search quality, retrieval/ranking, PyTorch, distributed training, representation learning, contrastive learning, multilingual/multimodal modeling, and RAG.
## Research question

> Can we build a search system that retrieves the most scientifically relevant geophysics evidence for a natural-language question, while improving retrieval with learned representations, hybrid sparse+dense search, ranking, and multimodal seismic information?

## Architecture

```text
Synthetic geophysics corpus
        |
        +--> BM25 sparse retrieval -------------------+
        |                                               |
        +--> Transformer dual encoder                   +--> Hybrid retrieval
        |       |                                       |
        |       +--> contrastive training               +--> Cross-encoder reranker
        |                                               |
        +--> Synthetic seismic waveform --> 1D CNN ----+
                                                        |
                                                        v
                                             Search evaluation
                                             Recall@k / MRR / nDCG
                                                        |
                                                        v
                                             RAG answer generation
```

## What this teaches relative to the job

| Job requirement | Project component |
|---|---|
| Search/retrieval | BM25, dense retrieval, hybrid retrieval |
| Retrieval quality | Recall@k, MRR, nDCG, hard-negative evaluation |
| PyTorch | Custom dual encoder + 1D CNN |
| Contrastive learning | In-batch InfoNCE loss |
| Representation learning | Query/document embedding space |
| Ranking | Cross-encoder reranking + learning-to-rank extension |
| Sparse + dense vectors | BM25 + transformer embeddings |
| Representation fusion | Text + seismic waveform embeddings |
| Multilingual | English/German query alignment experiment |
| RAG | Retrieve evidence, then generate grounded answer |
| Distributed training | `torchrun` training entry point |
| Optimization | batching, mixed precision, embedding caching |
| Research mindset | ablations and controlled experiments |
| Robust evaluation | fixed query set and regression report |

## Geophysics domain

The corpus is synthetic but scientifically themed around:
- induced seismicity
- pore-pressure diffusion
- fault reactivation
- effective stress
- acoustic/seismic emissions
- permeability evolution
- injection rate
- stress localization
- fault roughness
- seismic velocity
- hydraulic stimulation

The data generator deliberately creates **near-confusable documents**. For example, several papers can mention pore pressure, but only one may specifically discuss diffusion timescales. This makes retrieval evaluation more meaningful than simple keyword matching.

## Project stages

### Stage 1 — Baseline search

```bash
python -m data.generate_corpus
python -m retrieval.build_index
python -m evaluation.evaluate_retrieval
```

Compare:
1. BM25
2. dense retrieval
3. hybrid retrieval

### Stage 2 — Train your own representation model

```bash
python -m training.train_dual_encoder
python -m retrieval.build_index --encoder artifacts/dual_encoder
python -m evaluation.evaluate_retrieval
```

The dual encoder learns:

```text
query -> Transformer -> vector
                              \ cosine similarity
                               > relevant document
                              /
document -> Transformer -> vector
```

Training uses positive query-document pairs and in-batch negatives.

### Stage 3 — Ranking

Run a cross-encoder reranker over the top 20 retrieved documents. Compare retrieval-only vs reranked results.

### Stage 4 — Multimodal representation fusion

```bash
python -m models.train_waveform_encoder
python -m evaluation.evaluate_multimodal
```

A 1D CNN learns a compact representation of synthetic seismic waveforms. Fuse text and waveform embeddings and measure whether retrieval improves for waveform-aware queries.

### Stage 5 — RAG

```bash
python -m rag.answer "How does pore-pressure diffusion affect fault reactivation?"
```

The answer generator receives only retrieved evidence and is instructed to cite document IDs.

### Stage 6 — Research experiments

Run the ablations:

```bash
python -m evaluation.ablation
```

Suggested experiments:
- BM25 vs dense vs hybrid
- frozen encoder vs contrastively trained encoder
- 1 negative vs in-batch negatives
- top-k = 5/10/20
- reranker on/off
- text-only vs text+waveform fusion
- random negatives vs hard negatives
- English-only vs English+German queries

## Mac / CPU note

This project is designed to run locally. On Apple Silicon, PyTorch can use MPS where supported. The training scripts automatically choose MPS, CUDA, or CPU.

The first Transformer run downloads model weights. A practical starting model is `distilbert-base-uncased`.

If `faiss-cpu` does not have a compatible wheel for your Python version, remove it from `requirements.txt`: the baseline project can use NumPy cosine search instead.

## Distributed training

The code includes a small DDP-compatible entry point:

```bash
torchrun --standalone --nproc_per_node=2 -m training.train_ddp
```

On a normal laptop, start with one process. The goal is to understand the architecture rather than reproduce Perplexity-scale training.

## Evaluation philosophy

Do not judge the project by whether the model produces a nice-looking answer. Treat search quality as the primary research target.

Track:
- Recall@1, @5, @10
- MRR
- nDCG@10
- latency p50/p95
- embedding throughput
- reranking cost
- answer grounding / citation correctness

Keep a fixed evaluation set and do not repeatedly tune on the test queries.

## Interview-style research questions

1. Why can BM25 outperform a dense retriever on rare scientific terminology?
2. Why does contrastive learning improve retrieval?
3. What is the difference between a bi-encoder and a cross-encoder?
4. Why do in-batch negatives make training efficient?
5. What happens if the training negatives are too easy?
6. How would you mine hard negatives from search logs?
7. How would you prevent retrieval leakage between train and test?
8. How would you evaluate multilingual retrieval?
9. How would you fuse waveform and text representations?
10. What would you optimize first if p95 latency were too high?
11. How would you scale training from one GPU to 100 GPUs?
12. When would sparse retrieval be preferable to dense retrieval?
13. How would you diagnose a retrieval regression after changing the encoder?
14. How would you evaluate RAG separately from retrieval quality?
15. How would you design an online search experiment without contaminating the offline test set?

## Portfolio deliverable

The strongest version of this project is not the code alone. Produce a 4–6 page research report containing:

1. problem definition
2. dataset construction
3. baseline results
4. contrastive-learning method
5. ablation table
6. latency/throughput measurements
7. failure analysis with example queries
8. multimodal experiment
9. RAG grounding evaluation
10. what you would change at production scale

That structure mirrors the research loop: **hypothesis -> experiment -> metric -> failure analysis -> next hypothesis**.

### Optional Stage 7 — multilingual retrieval

```bash
python -m evaluation.evaluate_multilingual
```

This uses a multilingual pretrained encoder as a baseline. A stronger research version would train a multilingual dual encoder with aligned English/German query-document pairs.

### Optional Stage 8 — cross-encoder reranking

Use `retrieval/reranker.py` on the top 20 candidates from hybrid retrieval. Compare the quality gain against the extra inference latency. The cross-encoder jointly processes the query and candidate text, unlike the bi-encoder, which embeds them independently.

The cross-encoder is intentionally a second-stage model rather than the first-stage retriever.
