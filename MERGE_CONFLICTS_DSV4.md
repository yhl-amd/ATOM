# Merge into `main`: DSV4 data-plane integration notes

> **Status**: PR #6 (`yhl-amd:cursor/dsv4-kv-cache-reorg-6d88` → `ROCm/ATOM:main`)
> is `CONFLICTING`. The control-plane reorg merges cleanly; the conflicts are
> confined to the DSV4 attention data plane and are a real feature integration,
> not a mechanical merge. This doc records the exact divergence and a porting
> checklist so it can be done by someone who owns both sides, with GPU validation.
>
> Captured against `origin/main` at merge-base `2c682dc4` (branch was ~21 behind /
> ~30 ahead when written; re-verify the base before acting).

## TL;DR

- **Root cause**: this branch (Phase 1) **extracted** `build_kv_cache_tensor`
  from `deepseek_v4_attn.py` into `atom/model_ops/attentions/dsv4/kv_bind.py`,
  from the *pre-fp4* version. `main` meanwhile **kept it inline and evolved it
  heavily** — FP4 indexer (#1709), a unified `quant_mode`, and a 2buff FP8 rope
  pool. So our `kv_bind.py` is behind main's binding logic.
- **Correct resolution = port main's evolution into our extracted structure**,
  reconcile `write_mode` (ours) vs `quant_mode` (main), keep the CSA snapshot,
  then GPU-validate. Do NOT pick-a-side or naive-union — the errors are silent
  and only surface on GPU (long-context FP8/FP4 indexer logits, see plan §5.2).

## Conflict inventory (4 files)

### Mechanically resolvable (safe — resolutions verified)

**`atom/model_ops/v4_kernels/__init__.py`** — export lists diverged; take the
union. Verified the (cleanly-merged) `state_writes.py` actually defines all 7:
`capture_compressor_boundary(_reference)`, `restore_compressor_boundary(_reference)`,
`update_compressor_states`, `swa_write`, `swa_write_2buff_prepacked`. `__all__`
= union of both sides deduped, plus main's `FP4_MQA_BLOCK_K`,
`FP4_MQA_PARALLEL_UNIT_NUM`, `QKNormRopeOut`, `fp4_indexer_enabled`,
`csa_translate_pack(_reference)` and our `capture/restore_*` + `make_*_boundary_plan`.

**`atom/model_ops/attentions/gdn_attn.py`** — this branch moved
`from ...scheduler import ScheduledBatch` under `TYPE_CHECKING` (Phase 2, kills the
runtime reverse-import); main kept the runtime import. Keep **ours**: the file has
`from __future__ import annotations` and every `ScheduledBatch` is a parameter
annotation (lines ~287/303/313/615/628) — no runtime use, so the guarded import is
correct.

### Feature-level divergence (needs domain + GPU — do NOT blind-merge)

**`atom/models/deepseek_v4.py`** (Indexer/Compressor `__init__`)
- ours: `write_mode: str = "bf16"` (values `bf16` / `main_2buff_fp8` /
  `indexer_fp8`) **plus** CSA snapshot fields `boundary_kv`, `boundary_score`.
- main: `quant_mode: str = "none"` (values `none` / `group_fp8` / `per_row_fp8` /
  `fp4`) — a rename + expansion of the same compress-scatter mode field, with FP4.
- Decision needed: is `write_mode` fully subsumed by `quant_mode` (then drop it and
  migrate all readers) or do both coexist? `boundary_kv/score` **must be kept**.

**`atom/model_ops/attentions/deepseek_v4_attn.py`** (4 hunks)
1. Imports — ours adds `TYPE_CHECKING` + `kv_pool_layout` (KvLayoutOptions/
   KvPoolLayout) + `ScheduledBatch` guarded; main has minimal `from typing import
   Any, cast`. Union, keep our guarded imports.
2. `build_kv_cache_tensor` body — main added `unified_kv` / `unified_kv_rope`
   (2buff fp8 parallel rope pool) construction inline; **ours is empty here** because
   the body moved to `kv_bind.py`. → main's 2buff-rope construction must be ported
   into `kv_bind.py`.
3. The `_V4Indexer` / `_V4Compressor` binding block — main's inline version carries
   the **new FP4 logic**: `module._indexer_fp4`, `kv_scale = v4_csa_idx_kv_scale[pos]`,
   `module.cache_scale` FP4 vs FP8 branch, `module.quant_mode = "fp4"/"per_row_fp8"`.
   Ours delegates to `kv_bind.py` + explicit `CommonAttentionBuilder.build_kv_cache_tensor`
   fallback. → the FP4/`quant_mode` binding must be re-expressed inside `kv_bind.py`.
4. (remaining hunk) same family — inline binding vs extracted fallback.

## Porting checklist (the real work)

1. **Sync `kv_bind.py` to main's binding logic.** Re-extract main's *current*
   `build_kv_cache_tensor` (post-#1709) into the `_bind_v4_indexer` /
   `_bind_indexer_inner_compressor` / `_bind_main_compressor_c4|c128` helpers,
   including:
   - `_indexer_fp4` / `fp4_indexer_enabled` gating,
   - the FP4 scale pool (`v4_csa_idx_kv_scale[pos]`, `module.kv_scale`),
   - the 2buff FP8 rope pool (`unified_kv_rope`),
   - `quant_mode` assignment (`fp4` / `per_row_fp8` / `group_fp8` / `none`).
   Preserve the invariant our characterization test pins: FP8 `cache_scale`
   `as_strided(storage_offset = idx_kv_f32.storage_offset() + scale_fp32_offset)`
   (absolute offset — see kv_bind + plan §1.4). main's FP4 path binds a real
   per-pos tensor and is unaffected.
2. **Reconcile `write_mode` ↔ `quant_mode`.** Pick one field. If `quant_mode`
   supersedes `write_mode`, migrate the CSA-snapshot capture/restore path to read
   `quant_mode`; keep `boundary_kv`/`boundary_score`.
3. **CSA snapshot × FP4.** Confirm capture/restore boundary kernels behave under
   the FP4 indexer path (they were written for the FP8/bf16 world). If FP4 changes
   the indexer cache layout, the boundary snapshot slicing may need updating.
4. **Apply the two mechanical resolutions** above verbatim.
5. **`atom/utils/envs.py`** auto-merged — sanity-check no env flag was dropped.

## Validation (mandatory — CPU green is NOT sufficient)

```bash
rm -rf /root/.cache/atom/*
# 1. TP4 fp8 server boot, arena off/on × CSA snapshot off/on
# 2. TP? fp4 indexer path boot (gfx950) — the merged-in main feature
# 3. GSM8K forced-prefix-hit n=150   (catches wrong prefix reuse / CSA restore)
# 4. C128 long-context smoke         (catches storage_offset/stride silent misbind)
# 5. CSA boundary snapshot: python tests/test_csa_boundary_snapshot.py
```

## Alternative if this stalls

The control-plane reorg (kv_cache package, BlockManager split, renames, factory,
dead-code + naming cleanups) is independent of this data-plane divergence. If the
DSV4 integration needs to wait, consider splitting the Phase-1 `kv_bind` extraction
out of this stack and landing the control-plane work first, then integrating
`kv_bind` against current main separately.
