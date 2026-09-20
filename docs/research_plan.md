# Research plan

## Hypothesis 1
Dense representations should improve semantic retrieval when query wording differs from document wording.

**Test:** BM25 vs frozen Transformer vs contrastively trained dual encoder.

## Hypothesis 2
Hard negatives matter more than simply increasing the number of random negatives.

**Test:** choose top BM25 non-relevant documents as negatives and compare against random in-batch negatives.

## Hypothesis 3
Hybrid retrieval is more robust to scientific terminology than either sparse or dense retrieval alone.

**Test:** evaluate on rare terms, paraphrases, and short queries separately.

## Hypothesis 4
A second-stage cross-encoder can improve ranking quality but increases latency.

**Test:** measure nDCG gain against p50/p95 latency.

## Hypothesis 5
Multimodal evidence can disambiguate geophysical queries where text alone is insufficient.

**Test:** fuse waveform and text embeddings and create a subset of waveform-aware queries.

## Hypothesis 6
Multilingual alignment can transfer retrieval capability across English and German.

**Test:** train on paired English/German query-document examples and evaluate language-by-language.

## Scientific failure analysis

For every failed query, classify the failure as:
- lexical mismatch
- semantic mismatch
- wrong document granularity
- insufficient corpus coverage
- hard negative confusion
- multilingual mismatch
- ranking error
- data-label error

Do not only report a single average metric. A research engineer should explain *why* the system fails.
