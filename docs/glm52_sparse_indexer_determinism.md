# GLM-5.2 sparse-indexer deterministic decode top-k

## Problem

GLM-5.2 uses a replicated sparse indexer under tensor parallelism. Before
sparse MLA runs, every TP rank must select exactly the same logical KV token
positions.

On a 60,753-token recorded request, we observed byte-identical indexer inputs,
indexer cache writes, and paged DeepGEMM logits on all eight TP ranks. However,
the native AITER `top_k_per_row_decode` returned different local positions on
different ranks. Those ranks then attended to different KV subsets before
all-reduce, causing long-context output corruption.

### 因果图（中文）

```text
同一份 indexer logits（8 个 TP rank 逐字节一致）
                         │
                         ▼
       native top-k 未定义“同分时按哪个 token 选”
                         │
                         ▼
rank 0 ──> KV 集合 A: [101, 102, 103, 104]
rank 1 ──> KV 集合 B: [101, 102, 103, 105]
rank 2 ──> KV 集合 C: [101, 102, 103, 106]
                         │
                         ▼
       每张卡实际在计算不同 KV 子集上的 sparse MLA
                         │
                         ▼
                    tensor-parallel all-reduce
                         │
                         ▼
       A 的 attention 输出 + B 的输出 + C 的输出 + ...
                         │
                         ▼
          不再对应任何一个合法的 attention 结果
                         │
                         ▼
            后续层放大误差 → 标签泄漏、乱码、随机文本
```

例如 top-4 的边界有三个相同分数：

```text
token 位置:  101   102   103   104   105   106   107
logit:       9.2   8.7   8.1   7.5   7.5   7.5   6.9

原实现：第 4 名可在 104/105/106 中不确定地选择。
修复后：比较键为 (score 降序, token 位置升序)，全部 rank 固定选择 104。
```

The issue is confined to selection, not cache storage:

- the main MLA cache row and indexer cache's logical 132-byte entry were
  byte-identical across full prefill and two exact prefix-cache replays;
- the indexer cache uses a 16-token, 144-byte-stride preshuffled layout, and
  the writer passes/uses the physical 144-byte stride rather than the logical
  132-byte payload size;
- the paged logits were byte-identical while native top-k membership differed.

FP8 quantisation and the indexer's ReLU make equal-score boundaries common.
Without a secondary token-position key, a parallel top-k can choose different
members of such a boundary on independent rank launches.

## Fix

Set the opt-in environment variable before model construction:

```bash
ATOM_DETERMINISTIC_GLM_INDEXER_TOPK=1
```

For decode, this switches GLM sparse-indexer selection from AITER
`top_k_per_row_decode` to `_deterministic_top_k_per_row_decode`.

For each query row the fallback:

1. gets the kth score threshold with `torch.topk`;
2. keeps all positions with a score strictly greater than that threshold;
3. fills remaining positions among equal scores in ascending logical-token
   order.

The resulting comparison rule is `(score descending, token position ascending)`
and therefore has rank-stable membership. It avoids a full stable sort.

## vLLM NVIDIA native top-k reference

vLLM's CUDA sparse-indexer path is a useful performance reference, but its
tie-breaking semantics must be distinguished from this fix's strict TP
requirement. Its decode dispatcher has three relevant paths:

```text
CUDA and topK in {512, 1024, 2048}
  ├─ Hopper (SM90), <= 32 rows, aligned stride
  │    -> cooperative_topk: CTA cluster, TMA loads and DSMEM histogram
  ├─ other CUDA cases
  │    -> persistent_topk: persistent CTA / multi-CTA radix selection
  └─ other topK values or non-CUDA
       -> generic top_k_per_row_decode
```

The CUDA-specialised kernels do not fully sort a context row. They first
histogram scores in a coarse FP16-derived bin space, identify the bin
containing the kth item, and then refine candidates with four radix-256 passes
over the complete ordered FP32 score bits. This is a good design to retain for
the production AITER implementation: it avoids a full sort while resolving
scores that collide in a coarse FP16 bin.

For short/medium rows, vLLM has an explicit small-candidate comparison:

```cpp
(a.score > b.score) || (a.score == b.score && a.idx < b.idx)
```

This is the same desired ordering as this fix. Its CUDA code can be found in
vLLM's `sparse_attn_indexer.py`, `topk.cu`, `persistent_topk.cuh`,
`cooperative_topk.cuh`, and `topk_histogram_4096.cuh`.

However, this must not be read as a complete TP determinism guarantee for
every vLLM CUDA path. The large tie paths refine the 32 bits of the FP32 score,
but do not consistently include the logical token position as a secondary
radix key. Candidate collection/final exact-score selection still has forms
equivalent to:

```text
slot = atomicAdd(equal_score_counter, 1)
if slot < remaining:
    output[slot] = token_index
```

Thus, when the boundary contains more *bitwise-identical* FP32 scores than
remaining top-k slots, atomic arrival order can determine membership. The
score radix solves "same coarse bin but different FP32 score"; it does not by
itself solve "same FP32 score, choose the smallest logical positions". This is
particularly relevant here because FP8 quantisation followed by ReLU creates
many exact `0.0f` scores.

The intended AITER native follow-up therefore combines vLLM's efficient
score-radix structure with the missing final rule:

```text
1. Use histogram/radix passes to find the exact kth FP32 score.
2. Keep every position whose score is strictly greater.
3. For score == kth_score, select the smallest logical token positions until k.
```

Equivalently, the kernel must select by the lexicographic key
`(score descending, token position ascending)`. A position-aware
prefix-sum/radix pass (rather than an `atomicAdd` race) keeps that rule
graph-safe and makes independently launched TP ranks select the same KV set.

## Validation

With the fallback enabled on TP=8:

- all ranks selected the same local top-k positions;
- converted global indices and sparse MLA selected KV matched across ranks;
- full prefill plus two exact prefix-cache replays returned the same response
  content at `temperature=0`.

## Limitation and follow-up

The fallback reads dynamic context lengths on the host and is intentionally
eager-only. It should not be enabled for level-3 CUDAGraph serving.

The production solution is an AITER `top_k_per_row_decode` implementation that
compares score and token position inside its parallel reduction, so it remains
deterministic and graph-safe without this fallback.
