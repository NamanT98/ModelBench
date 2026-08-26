# Engineering Optimization: Inference Batching

## Context

The ModelBench project originally processed inference generation requests sequentially (batch size = 1). As we prepared to run the comprehensive M7 Hybrid Lexical + Semantic Retrieval experiments (which require running the full 1,034-sample Spider dev set multiple times), total experiment runtime became a critical bottleneck (~2 hours per run on an RTX 5060 8GB).

To make large-scale systematic experimentation practical, we implemented native HuggingFace batched generation, allowing multiple samples to be processed concurrently on the GPU without altering existing behavior, retrieval ranking, or prompt construction.

## Benchmark Methodology

We benchmarked the batching implementation against the deterministic baseline using a sample size of 50 Spider dev queries on an RTX 5060 (8GB VRAM). The evaluation focused on ensuring byte-for-byte evaluation equivalence, monitoring VRAM limits, and determining the optimal stable throughput.

### Configuration
- **Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Retrieval Strategy**: Zero-shot (to isolate generation time)
- **Prompt Strategy**: Normalized schema linking
- **Decoding**: Greedy (`do_sample=False`, `temperature=0.0`)
- **Device**: CUDA (`float16`)

## Results

| Batch | Runtime (s)  | Samples/s | Peak VRAM (MB)  | Eq. Valid  | Eq. Exec Acc |
|-------|--------------|-----------|-----------------|------------|--------------|
| 1     | 151.17       | 0.33      | 6058.81         | N/A        | N/A          |
| 8     | 95.43        | 0.52      | 6294.88         | Yes        | Yes          |
| 12    | 87.71        | 0.57      | 6426.35         | Yes        | Yes          |
| **16**| **87.43**    | **0.57**  | **6521.30**     | **Yes**    | **Yes**      |
| 32    | 92.04        | 0.54      | 6982.41         | Yes        | Yes          |

## Key Findings

1. **Validation & Integrity**:
   - Extraction validity and execution accuracy are **100% equivalent** across all batch sizes when compared to the `batch_size=1` baseline. Left-padding and output slicing logic was implemented correctly and respects per-sample lengths.
   - Exact-match outcomes (implied by execution parity) are functionally identical for all test samples.

2. **Speedup**:
   - The optimal batch size of 16 reduced the 50-sample runtime from ~151s to ~87s.
   - **Performance Gain**: **~1.75x speedup**. This reduces the latency of a full 1,034-sample run from roughly 2 hours to approximately 1 hour and 10 minutes.
   - We observed diminishing returns past batch size 12, with batch size 32 actively degrading performance compared to batch sizes 12 and 16 due to padding inefficiency overhead.

3. **Memory Profile**:
   - The `Qwen2.5-Coder-3B-Instruct` float16 model takes around ~5.8 GB at rest.
   - Batch size 1 peaks at 6,058 MB VRAM.
   - Batch size 16 peaks at 6,521 MB VRAM. 
   - Batch size 32 peaks at 6,982 MB VRAM.
   - Since the RTX 5060 has 8GB VRAM, batch size 16 leaves a ~1.5 GB safety margin, ensuring strict OOM safety while providing maximum throughput.

## Conclusion

Batching is purely an engineering optimization required to make large-scale experimentation feasible on constrained hardware. We found `batch_size=8` or `16` to be optimal for 8GB VRAM configurations, delivering identical experimental outcomes in significantly less time.
