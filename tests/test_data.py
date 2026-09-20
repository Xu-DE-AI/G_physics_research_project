from pathlib import Path

def test_project_files():
    for p in ["README.md","requirements.txt","data/generate_corpus.py","training/train_dual_encoder.py","evaluation/evaluate_retrieval.py"]:
        assert Path(p).exists()
