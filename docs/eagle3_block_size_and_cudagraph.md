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

## 3. Phase 2：启用 Eagle3 CUDAGraph（target model）

**状态**：已完成

### 背景

MTP 的 cudagraph 仅覆盖 target model（drafter 始终 eager），Eagle3 被两处 `eagle3_mode` 检查硬编码跳过，导致 target model 也走 eager，性能白白损失。

### 改动要点

| 改动 | 说明 |
|---|---|
| 删除 `capture_cudagraph()` 的 `eagle3_mode` 跳过 | 原先直接 return 空 graphs，现在正常捕获 |
| 删除 `run_model()` 的 `eagle3_mode` eager 强制 | 原先 `or self.eagle3_mode` 阻止走 graph replay |
| `capture_cudagraph()` 处理 tuple 返回值 | Eagle3 开启 `aux_hidden_state` 后 model forward 返回 `(hidden_states, aux_hidden_states)` 元组，需拆开赋值 |
| 新增 `graph_aux_hidden` 字典 | 按 `(graph_bs, max_q_len)` 保存 graph capture 内的 aux tensor 引用，replay 后原地更新 |
| `run_model()` replay 后恢复 `_aux_hidden_states` | 从 `graph_aux_hidden` 取出 aux 供 Eagle3 drafter 使用 |

### tuple 返回值问题

MTP 不用 `aux_hidden_states`，model forward 返回单个 tensor，直接 `outputs[:] = self.model(...)` 即可。Eagle3 配置了 `eagle3_aux_layer_ids`（layer 1, 29, 57），model forward 返回 `(hidden_states, [aux1, aux2, aux3])`，直接赋值报 `TypeError: can't assign a tuple to a torch.cuda.BFloat16Tensor`。

修复：warmup 和 graph capture 中用 `isinstance(model_output, tuple)` 拆包。`isinstance` 是类型检查而非数据依赖分支，model 返回类型在 capture 时已确定，不影响 cudagraph 捕获。

### 待验证

- [ ] 去掉 `--enforce-eager` 后 cudagraph capture 成功
- [ ] decode 阶段走 graph replay（日志中应出现 `decode[...]` 而非 `prefill[...]`）
- [ ] Eagle3 drafter 在 graph replay 后能正确拿到 `_aux_hidden_states`
- [ ] 与 `--enforce-eager` 对比输出一致性和性能提升

### 后续可选

- **drafter cudagraph**：MTP 的 drafter 也是 eager，可独立实现（预分配 buffer + 按 graph_bs 捕获）
- **drafter torch.compile**：将 `NO_COMPILATION` 提升到 `PIECEWISE`（需解决 KV 绑定与 Dynamo 时序问题）

### 风险

- 回退方案：`--enforce-eager` 恢复全 eager
- 显存影响可忽略：Eagle3 只有 1 层 + GQA=1，以 10K blocks、bf16 计算增量约 3.6 MB
