import time
import json
from pathlib import Path

from modelbench.config import load_config, DatasetConfig
from modelbench.dataset import SpiderDataset
from modelbench.retrieval import create_retriever

def main():
    # Load dataset
    print("Loading Spider Dev Split (first 50 samples)...")
    dev_config = DatasetConfig(name="spider", path="datasets/spider", split="dev")
    dev_set = list(SpiderDataset(dev_config).load())[:50]
    
    train_config = DatasetConfig(name="spider", path="datasets/spider", split="train")
    train_set = list(SpiderDataset(train_config).load())
    
    candidate_ns = [25, 50, 100, 200, None]
    
    results = {}
    
    for n in candidate_ns:
        print(f"\nInitializing Retriever with candidate_n={n}...")
        retriever = create_retriever(
            "hybrid_rrf",
            train_set,
            embedding_model="BAAI/bge-small-en-v1.5",
            hybrid_rrf_constant=60,
            hybrid_candidate_n=n
        )
        
        start_time = time.perf_counter()
        
        overlap_counts = []
        outputs = []
        for sample in dev_set:
            res = retriever.retrieve(sample.question, k=3)
            outputs.append([s.question for s in res.samples])
            
        latency = (time.perf_counter() - start_time) / len(dev_set)
        
        results[n] = {
            "latency": latency,
            "outputs": outputs
        }
        print(f"candidate_n={n}: Avg Latency: {latency:.4f}s")
        
    print("\nSanity Check vs None (Full RRF):")
    ref_outputs = results[None]["outputs"]
    for n in [25, 50, 100, 200]:
        n_outputs = results[n]["outputs"]
        exact_matches = sum(1 for a, b in zip(n_outputs, ref_outputs) if a == b)
        print(f"candidate_n={n}: Exact Top-3 Match: {exact_matches}/{len(dev_set)} ({(exact_matches/len(dev_set))*100:.1f}%)")

if __name__ == "__main__":
    main()
