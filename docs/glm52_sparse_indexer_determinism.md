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
