# From laptop project to search production

| Laptop | Production-scale question |
|---|---|
| CSV corpus | How do we ingest fresh documents and search logs continuously? |
| NumPy/BM25 | Elasticsearch/OpenSearch or equivalent sparse index |
| Transformer encoder | GPU inference service + embedding cache |
| Local vectors | FAISS / vector database / distributed ANN |
| PyTorch DDP | multi-node training, FSDP/DeepSpeed, checkpoint sharding |
| Cross-encoder | GPU reranker with batching and dynamic top-k |
| Fixed evaluation JSON | continuously maintained benchmark + regression gates |
| FastAPI | scalable model/search service |
| printed metrics | dashboards for quality, latency, throughput and cost |
| local RAG prompt | grounded generation service with citation checks |

## Scaling questions to practice

### Training
- Data parallelism vs model parallelism
- gradient accumulation
- mixed precision
- activation checkpointing
- FSDP vs DDP
- checkpoint size and recovery

### Retrieval
- approximate nearest-neighbor search
- HNSW / IVF / PQ
- embedding dimensionality
- index refresh strategy
- query caching
- multilingual indexes

### Ranking
- pointwise vs pairwise vs listwise objectives
- hard-negative mining
- click bias and position bias
- offline/online metric mismatch

### Evaluation
- fixed benchmark leakage
- query stratification
- long-tail queries
- multilingual slices
- latency p50/p95/p99
- quality/latency tradeoff

### RAG
- retrieval recall vs answer faithfulness
- citation correctness
- context-window budget
- retrieval failure vs generation failure
