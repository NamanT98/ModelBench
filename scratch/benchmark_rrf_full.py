import json
import subprocess
import pandas as pd
from pathlib import Path
import os
import yaml

def run_experiment(candidate_n, name):
    print(f"\n--- Running Experiment: {name} (candidate_n={candidate_n}) ---")
    
    # Create config file
    config_path = f"configs/m7_hybrid_rrf_n{candidate_n if candidate_n is not None else 'None'}.yaml"
    config = {
        "experiment": {"name": name},
        "dataset": {"name": "spider", "path": "datasets/spider", "split": "dev", "limit": None},
        "model": {"provider": "huggingface", "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct"},
        "generation": {"max_new_tokens": 256, "temperature": 0.0, "batch_size": 8},
        "schema": {"strategy": "schema_linking_normalized"},
        "strategy": {
            "name": "few_shot",
            "retriever": "hybrid_rrf",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "k": 3,
            "hybrid_rrf_constant": 60,
            "hybrid_candidate_n": candidate_n
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Run the experiment
    cmd = ["modelbench", "run", "--config", config_path]
    
    # Run the command and print output to stdout
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error running experiment {name}")
    
    return f"results/{name}.json"

def analyze_results(files, ref_file):
    # Load reference results
    with open(ref_file) as f:
        ref_data = json.load(f)
        ref_samples = ref_data["samples"]
        
    results = []
    
    for name, path in files.items():
        try:
            with open(path) as f:
                data = json.load(f)
                
            metrics = {
                "Candidate N": name,
                "Exec Acc": data.get("execution_accuracy", 0) * 100,
                "SQL Validity": data.get("sql_validity_rate", 0) * 100,
                "Exact Match": data.get("exact_match_rate", 0) * 100,
                "Retrieval Latency (s)": 0.0,
                "Total Latency (s)": data.get("avg_latency_seconds", 0)
            }
            
            # Compute average retrieval latency from diagnostics
            retrieval_latencies = [s.get("retrieval", {}).get("diagnostics", {}).get("latency_seconds", 0) for s in data["samples"]]
            if retrieval_latencies:
                metrics["Retrieval Latency (s)"] = sum(retrieval_latencies) / len(retrieval_latencies)
                
            # Compute overlaps and diffs against reference
            exact_top3 = 0
            top3_overlap_sum = 0
            exec_diffs = 0
            
            for s_opt, s_ref in zip(data["samples"], ref_samples):
                opt_ids = s_opt.get("retrieval", {}).get("diagnostics", {}).get("hybrid_top_k_ids", [])
                ref_ids = s_ref.get("retrieval", {}).get("diagnostics", {}).get("hybrid_top_k_ids", [])
                
                # We can just compare questions since they are deterministic
                if opt_ids == ref_ids:
                    exact_top3 += 1
                
                overlap = len(set(tuple(x) for x in opt_ids) & set(tuple(x) for x in ref_ids))
                top3_overlap_sum += (overlap / 3.0) if len(ref_ids) == 3 else 1.0
                
                if s_opt.get("evaluation", {}).get("execution_match") != s_ref.get("evaluation", {}).get("execution_match"):
                    exec_diffs += 1
                    
            n_samples = len(ref_samples)
            metrics["Exact Top-3 Match %"] = (exact_top3 / n_samples) * 100
            metrics["Top-3 Overlap %"] = (top3_overlap_sum / n_samples) * 100
            metrics["Outcome Diffs %"] = (exec_diffs / n_samples) * 100
            
            results.append(metrics)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            
    df = pd.DataFrame(results)
    df = df[["Candidate N", "Exec Acc", "SQL Validity", "Exact Match", "Retrieval Latency (s)", "Total Latency (s)", "Exact Top-3 Match %", "Top-3 Overlap %", "Outcome Diffs %"]]
    print("\n--- Final Results ---\n")
    print(df.to_markdown(index=False, floatfmt=".2f"))

def main():
    candidate_ns = [25, 50, 100, 200]
    
    files = {}
    for n in candidate_ns:
        name = f"spider_qwen_m7_hybrid_rrf_n{n}"
        if not os.path.exists(f"results/{name}.json"):
            files[n] = run_experiment(n, name)
        else:
            files[n] = f"results/{name}.json"
            print(f"Skipping {n} as it already exists at results/{name}.json")
            
    # Reference is the original full run
    ref_file = "results/spider_qwen_m7_hybrid_rrf.json"
    files["Full (None)"] = ref_file
    
    analyze_results(files, ref_file)

if __name__ == "__main__":
    main()
