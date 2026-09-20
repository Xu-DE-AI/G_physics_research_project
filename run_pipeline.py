import subprocess, sys

def run(cmd):
    print("\n$", " ".join(cmd)); subprocess.run([sys.executable, *cmd], check=True)

run(["-m","data.generate_corpus"])
run(["-m","evaluation.evaluate_retrieval"])
run(["-m","evaluation.ablation"])
print("\nPipeline complete. Next: python -m training.train_dual_encoder")
