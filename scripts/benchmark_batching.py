import copy
import time
import torch
from pathlib import Path
from modelbench.config import load_config
from modelbench.runner import ExperimentRunner

def main():
    # Load default modelbench config to get valid dataset paths
    config = load_config()
    
    # Override for our benchmark setup
    config.dataset.limit = 50
    config.dataset.path = "datasets/spider"
    config.model.model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    config.model.device = "cuda"
    config.model.dtype = "float16"
    config.generation.max_new_tokens = 256
    config.generation.temperature = 0.0
    config.generation.do_sample = False
    
    # Use deterministic retrieval-free baseline for Generation Benchmark
    config.schema.strategy = "schema_linking_normalized"
    config.strategy.name = "zero_shot"

    batch_sizes = [1, 8, 12, 16, 32]
    baseline_results = None
    
    print("Starting Benchmark...")
    print(f"{'Batch':<5} | {'Runtime (s)':<12} | {'Samples/s':<10} | {'Peak VRAM (MB)':<15} | {'Eq. Valid':<10} | {'Eq. Exec Acc':<12}")
    print("-" * 75)

    for bsz in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Deep copy to ensure clean state
            run_config = copy.deepcopy(config)
            run_config.generation.batch_size = bsz
            
            runner = ExperimentRunner(run_config)
            
            start_time = time.perf_counter()
            result = runner.run()
            total_time = time.perf_counter() - start_time
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            samples_per_sec = result.total_samples / total_time
            
            # Extract lists for equivalence checking
            validity_list = [s.sql_valid for s in result.samples]
            exec_acc_list = [s.execution_accuracy for s in result.samples]
            
            if bsz == 1:
                baseline_results = {
                    "validity": validity_list,
                    "exec_acc": exec_acc_list,
                }
                eq_valid = "N/A"
                eq_exec = "N/A"
            else:
                eq_valid = "Yes" if validity_list == baseline_results["validity"] else "No"
                eq_exec = "Yes" if exec_acc_list == baseline_results["exec_acc"] else "No"
                
            print(f"{bsz:<5} | {total_time:<12.2f} | {samples_per_sec:<10.2f} | {peak_vram:<15.2f} | {eq_valid:<10} | {eq_exec:<12}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"{bsz:<5} | {'OOM':<12} | {'N/A':<10} | {'N/A':<15} | {'N/A':<10} | {'N/A':<12}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{bsz:<5} | {'ERROR':<12} | {str(e)[:10]:<10} | {'N/A':<15} | {'N/A':<10} | {'N/A':<12}")
            raise e

if __name__ == "__main__":
    main()
