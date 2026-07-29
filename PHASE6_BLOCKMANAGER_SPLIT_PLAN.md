# Phase 6 完成计划：把 DSV4 lifecycle 从 BlockManager 剥离

> 目标：`atom/model_engine/block_manager.py` 变成**真·通用** dense manager —
> `grep -i "csa\|swa\|arena\|compress\|boundary\|dsv4"` 结果为 0；全部 DSV4
> arena/SWA/CSA 逻辑迁入 `atom/kv_cache/dsv4/dsv4_kv_cache_manager.py`。

## 为什么现在能干净切

1. **dense 今天已靠 Null-Object 工作**：`Dsv4SwaPool(num_blocks=0)` → `enabled=False`，
   所有方法早返回（`has_free→True`、`bounded_hit→compressed_hit`、其余 mutator no-op）。
   → 这给出每个 base 钩子的**精确 no-op 默认值**，不是我猜的。
2. **外部 reach-through 极少**：只有测试碰 `_has_free_compressed`/`_evict_cold_compressed`/
   `free_block_ids_set`（都在 Dsv4 实例上，随方法搬到子类仍可用）和 `swa_enabled`
   （dense+dsv4 都要 → base 保留返回 False）。`requires_csa_boundary_state` 零外部读者。
3. Scheduler 已走 Protocol，不 reach-through `.swa`/`.arena`（前面审计确认）。

## 采用方案：模板方法 seam（单一事实来源）

base `BlockManager` 保留通用算法骨架，DSV4 专属处替换为**中性名钩子**（base 全 no-op，
Dsv4 子类 override）。这样 base 无 csa/swa/arena 字面量，且**通用分配算法只有一份**
（stale-hash 淘汰、ref 计数、cache-hit-on-free-pool claim 这些微妙逻辑不复制、不发散）。

### 钩子清单（base 默认 = 现 disabled-swa 语义）

| 钩子 | base 默认 | Dsv4 override |
|---|---|---|
| `_logical_capacity(config)` / `_arena_on` | `num_kvcache_blocks` / False | 先建 arena，返回 `arena.max_compressed_blocks()` |
| `_init_sidecars(config)` | pass | 建 `self.swa`、`_require_csa_boundary_state`、`attach_arena` |
| `_back_block(block_id)`（在 `_allocate_block`） | pass | `_arena_alloc_compressed` |
| `_primary_has_free(n)`（在 `can_allocate`/`can_append`） | `len(free_block_ids_set) >= n` | `_has_free_compressed(n)` |
| `_bounded_window_hit(seq, hit, hashes)` | 返回 `hit` | `swa.bounded_hit(...)` |
| `_window_has_free(seq, n)` | True | `swa.has_free(min(n, swa.admission_blocks(seq)))` 等 |
| `_window_claim_cached(seq, i, h, toks, hit)` | pass | swa_hit_start + `alloc_placeholder`/`claim_cached` |
| `_window_alloc_new(seq)` | pass | `swa.alloc_placeholder` |
| `_set_boundary_state(seq, hit)` | pass | CSA boundary block 计算 |
| `_publish_window_hash(seq, i, h, toks)` | pass | `swa.publish_hash` |
| `_release_window(seq)` | pass | `swa.release` + `seq.csa_boundary_state_block_id=-1` |
| `_window_append_new(seq)` / `_window_free_out(seq, n)` | pass | `swa.append_new` / `swa.free_out_of_window` |
| `swa_enabled` (property) | False | `self.swa.enabled` |
| `build_batch_tables(seqs)` | `HybridKvCacheTables.empty(...)` | `build_dsv4_batch_tables(...)` |
| `materialize_window`/`ensure_window_for_tokens`/`finish_prefill_chunk` | pass | swa 委托 |

### 整体搬入 Dsv4 子类的方法（base 删除）

`_build_arena`、`_evict_cold_for_borrow`、`_evict_cold_compressed`、
`_arena_alloc_compressed`、`_has_free_compressed`、`requires_csa_boundary_state`。

### base 删除的 import

`Dsv4UnifiedArena`、`build_dsv4_batch_tables`、`Dsv4SwaPool`、`ArenaEmpty`
（`HybridKvCacheTables` 保留 —— 它是中性的，base 空表要用）。

## 不动（红线）

- 不改 `v4_kernels/` 语义、不改 arena 借还策略、不改 kv_bind（storage_offset 那类静默 bug 源头不碰）。
- dense 分配行为**逐字节不变**（base 骨架 + no-op 钩子 == 现 dense 路径）。
- DSV4 分配行为不变（钩子 override == 现内联逻辑）。

## 验证

**CPU 门禁（我负责跑绿）**：
```
tests/test_block_manager.py            # dense: BlockManager 直接构造
tests/test_block_manager_arena.py      # dsv4: arena 借还 + ID 守恒
tests/test_scheduler.py tests/test_prefill_scheduler.py
tests/test_kv_cache_manager_contract.py tests/test_kv_events.py
tests/test_per_req_cache_decoupling.py tests/test_unified_kv_arena.py
+ ruff/black
```

**GPU 门禁（必须你在 GPU 环境验，CPU 绿≠正确 —— §5.2）**：
- TP4 fp8 boot（arena off/on × CSA snapshot off/on）
- GSM8K forced-prefix-hit n=150（这是抓「swa/csa claim 错→prefix 复用错」的唯一 oracle）
- C128 长上下文 smoke

## 风险与备选

- **风险**：碰核心分配路径；错一个钩子 = 错误 prefix 复用，CPU 可能静默、GPU GSM8K 才现形。
- **备选 S-B（方法复制）**：base=dense 版、Dsv4 全量 override `allocate`/`can_allocate` 等。
  各类自包含好读，但 ~120 行通用骨架复制两份，未来改一处要改两处（发散风险）。
  → 我推荐 seam（S-A），除非你偏好自包含可读性。

## 提交/PR

- 独立 commit，续到 PR #6（base=main）。
- 与 PR #6 那 3 个 DSV4 merge 冲突是两码事，互不阻塞。
