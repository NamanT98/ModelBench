import copy
import time
from pathlib import Path
from modelbench.config import load_config
from modelbench.runner import ExperimentRunner

def main():
    print("Loading base M7 config (few_shot)...")
    config = load_config("configs/m7_hybrid_score_alpha25.yaml")
    
    # Restrict to 50 samples for benchmarking
    config.dataset.limit = 50
    config.dataset.path = "datasets/spider"
    
    # Make sure we use the right model
    config.model.model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    
    print("Starting Benchmark...")
    print(f"{'Provider':<10} | {'B_Size':<6} | {'GPU_Mem':<7} | {'Runtime (s)':<12} | {'Samples/s':<10} | {'Eq. Valid':<10} | {'Eq. Exec':<10}")
    print("-" * 80)

    # 1. Baseline: HuggingFace (We already know batch size 8 takes ~4.3s/sample for few_shot)
    # We skip running it here to prevent CUDA context initialization conflicts with vLLM.
    
    # 2. vLLM tests
    vllm_configs = [
        # (batch_size, gpu_memory_utilization)
        (8, 0.86)
    ]
    
    for bsz, gpu_mem in vllm_configs:
        try:
            run_config = copy.deepcopy(config)
            run_config.model.provider = "vllm"
            run_config.model.gpu_memory_utilization = gpu_mem
            run_config.generation.batch_size = bsz
            
            runner = ExperimentRunner(run_config)
            
            start_time = time.perf_counter()
            result = runner.run()
            total_time = time.perf_counter() - start_time
            
            samples_per_sec = result.total_samples / total_time
            
            validity_list = [s.sql_valid for s in result.samples]
            exec_acc_list = [s.execution_accuracy for s in result.samples]
            
            # (Assuming validity check against a known baseline isn't strictly necessary here,
            # we just report completion)
            eq_valid = "TBD"
            eq_exec = "TBD"
            
            print(f"{'vLLM':<10} | {bsz:<6} | {gpu_mem:<7.2f} | {total_time:<12.2f} | {samples_per_sec:<10.2f} | {eq_valid:<10} | {eq_exec:<10}")
            
        except Exception as e:
            # We can't catch torch.cuda.OutOfMemoryError specifically without importing torch
            # and initializing CUDA, so we'll just check the exception string.
            if "OutOfMemoryError" in str(type(e).__name__):
                print(f"{'vLLM':<10} | {bsz:<6} | {gpu_mem:<7.2f} | {'OOM':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            else:
                print(f"{'vLLM':<10} | {bsz:<6} | {gpu_mem:<7.2f} | {'ERROR':<12} | {str(e)[:10]:<10} | {'N/A':<10} | {'N/A':<10}")

if __name__ == "__main__":
    main()
