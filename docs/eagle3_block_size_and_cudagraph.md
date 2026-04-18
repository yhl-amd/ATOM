# Eagle3 block_size 修复与 CUDAGraph 支持

## 1. 问题背景

Eagle3 draft 模型使用标准 MHA attention，但 KV cache 分配继承了 MLA 的 `physical_block_size=1`（来自 `AiterMLAMetadataBuilder`）。这导致：

1. **block_size=1 打崩所有 MHA decode kernel**：`pa_decode_gluon` 要求 block_size>=16，CK `fmha_v3` 和 triton gather 要求 16 元素对齐
2. **被迫写纯 PyTorch SDPA fallback**（`eagle3_decode_attention`，86 行），绕开所有 kernel
3. **SDPA fallback 堵死 cudagraph**：动态形状 gather、运行时 mask、host-side `.item()` 同步，全部不可捕获
4. **`model_runner.py` 硬编码跳过 cudagraph**：`capture_cudagraph()` 和 `run_model` 中 `eagle3_mode` 直接走 eager

根因链：`block_size=1` 继承 → kernel 崩溃 → SDPA fallback → cudagraph 不可用

### MLA 的 block_size=1 为什么合理

MLA 将 K/V 压成 576 维 latent，每 token KV 仅 ~1152B。AITER MLA 采用双粒度设计：寻址按 `page_size=1`（token 级灵活度、零内部碎片、精确 spec decode 回滚），kernel 计算按 `kv_granularity=16`（tile FMA 带宽不丢）。这对 MLA 自身合理，但 Eagle3 的 MHA kernel 不支持这种粒度。

---

## 2. Phase 1：修复 Eagle3 KV cache block_size

**状态**：已完成（+20 行，-133 行）

### 核心思路

给 Eagle3 独立的 `eagle3_block_size=16`，复用调度器已有的 `block_tables`（已在 `forward_vars` 中），让 Eagle3 走 `paged_attention_triton` → `pa_decode_gluon` 路径。

### 关键发现

- **`slot_mapping` 在 MLA 和 MHA builder 中构造方式完全一致**，不是 MLA 特有的
- **`forward_vars["block_tables"]` 在 MLA 路径也被填充**（`prepare_block_tables()` 基类方法），Eagle3 可直接使用
- **i=0 多 token 问题不存在**：`pa_decode_gluon` 通过 `max_seqlen_q > 1` 原生支持，之前仅因 MLA metadata 不设 `block_tables` 才需要 SDPA fallback

### 改动要点

| 文件 | 改动 |
|---|---|
| `model_runner.py` | 新增 `eagle3_block_size`；KV cache 分配和绑定 6 处 `physical_block_size` → `eagle3_block_size` |
| `eagle.py` | propose 循环前从 `forward_vars` 直接设 `block_tables/context_lens`；删除 `.item()` / for-loop / `torch.zeros` 重建逻辑 |
| `attention_mha.py` | 删除 `eagle3_decode_attention`（86 行 SDPA fallback）；简化 dispatch 路由 |
| `eagle3_llama.py` | 保持 `per_layer_sliding_window=0`（仍需用于 triton 路由，避免 ASM kernel 崩溃） |

### OOM 修复

Phase 1 初版导致 Eagle3 KV cache 16 倍超额分配。原因：

- MLA 的 `block_ratio=16` 使 `num_physical_kvcache_blocks = num_kvcache_blocks × 16`
- Eagle3 用 `eagle3_block_size=16` 但块数沿用 `num_physical_kvcache_blocks`
- 总容量 = `num_kvcache_blocks × 16（block_ratio）× 16（block_size）= 256倍`，实际只需 `num_kvcache_blocks × 16`

修复：Eagle3 块数改为 `num_physical_kvcache_blocks // eagle3_block_size`（= `num_kvcache_blocks`），同步修正 `_kv_cache_bytes_per_block()` 的估算。涉及 `model_runner.py` 三处：`_kv_cache_bytes_per_block()`、`allocate_kv_cache()`、`bind_kv_cache()`。

**经验**：当子系统（Eagle3 MHA）的 KV cache 挂载在另一个子系统（MLA）的内存管理下，block 粒度转换必须双向一致——估算粒度和分配粒度不匹配会导致 OOM 或显存浪费。

### 待验证

- [ ] Eagle3 + Kimi K2.5 正常启动（`--enforce-eager`），prefill + decode 端到端正确出字
- [ ] k_cache shape：`[num_blocks, kv_heads, head_dim//x, 16, x]`
- [ ] fp8 kv_scale shape 正确
- [ ] 对比修改前后输出一致性（temperature=0，draft token 接受率）

---

## 3. Phase 2 计划：启用 Eagle3 CUDAGraph

**状态**：待实施（依赖 Phase 1 验证通过）

Phase 1 消除了 cudagraph 的所有阻塞因素：SDPA fallback 已删除、`.item()` 同步已删除、动态 `torch.zeros` 已删除、i=0 多 token 分支已消除。

### 计划步骤

1. **预分配 drafter buffer**：`draft_input_ids`、`draft_positions`、`draft_hidden_states` 按 `max_bs` 预分配
2. **i=0 走 eager，i>=1 走 cudagraph replay**：drafter 只捕获 `max_q_len=1` 的图，按 `graph_bs` 建若干张
3. **移除 target model 的 eagle3_mode cudagraph 跳过**：删除 `capture_cudagraph()` 的 `if self.eagle3_mode: return` 和 `run_model` 的 `or self.eagle3_mode`
4. **drafter 保持 `NO_COMPILATION`**：cudagraph 捕获 eager forward，后续可选 piecewise compile

### 风险

- Phase 1 风险低：block_size 值替换 + dispatch 简化，MTP 和 target model 路径不受影响
- Phase 2 风险中等：cudagraph 捕获涉及固定输入形状、graph pool 管理。回退方案：`--enforce-eager`
- 显存影响可忽略：Eagle3 只有 1 层 + GQA=1，以 10K blocks、bf16 计算增量约 3.6 MB
