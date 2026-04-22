# Eagle3 投机解码接受率分析报告

## 概述

本文档记录 ATOM 引擎 Eagle3 投机解码接受率从 ~1% 到 64.5% 的完整排查过程，以及与 vLLM 参考实现（72.9%）的对比分析。

## 测试环境

| 项目 | 配置 |
|------|------|
| Target 模型 | `Kimi-K2.5-MXFP4`（MXFP4 量化） |
| Draft 模型 | `kimi-k2.5-eagle3`（`Eagle3LlamaModel`，1 层 Llama decoder） |
| 硬件 | AMD MI350 × 8，ROCm 7.2 |
| TP | 8 |
| num_speculative_tokens | 3 |
| eagle3_aux_layer_ids | 1, 29, 57 |
| 评测 | lm_eval GSM8K, 5-shot, num_concurrent=64 |

## 指标汇总

### 接受率演进

| 阶段 | Acceptance Rate | Avg toks/fwd | GSM8K Accuracy | 日期 |
|------|:-:|:-:|:-:|:-:|
| 初始（两个 bug） | ~1% | ~1.01 | 93.03% | 2026-04-21 |
| Fix 1: dtype 转换 + aux 默认值 | 60.8% | 2.82 | 93.25% | 2026-04-21 |
| Fix 2: 消除双重 RMSNorm | 64.5% | 2.94 | 93.48% | 2026-04-22 |
| Fix 3: aux IDs `(2,30,58)` 对齐 vLLM 默认 | **65.9%** | **2.98** | **93.78%** | 2026-04-22 |
| vLLM 参考值 | **72.9%** | **3.19** | 92.65% | 2026-04-22 |
| 官方 benchmark（未量化模型） | — | 3.20 (GSM8K) | — | HuggingFace |

### ATOM 修复后 vs vLLM 详细对比

| 指标 | ATOM (Fix 3) | vLLM | 差距 |
|------|:-:|:-:|:-:|
| Acceptance Rate | 65.9% | 72.9% | -7.0% |
| Avg toks/fwd | 2.98 | 3.19 | -0.21 |
| 全部拒绝 (0 accepted) | 17.3% | — | — |
| 接受 1 token | 15.6% | — | — |
| 接受 2 tokens | 18.7% | — | — |
| 全部接受 (3 accepted) | 48.4% | — | — |
| Per-position (pos 1/2/3) | — | 87% / 73% / 59% | — |
| GSM8K Accuracy | 93.78% | 92.65% | +1.13% |

### vLLM 吞吐参考

| 指标 | vLLM |
|------|:-:|
| Avg prompt throughput | ~15000 tokens/s |
| Avg generation throughput | ~1900-2100 tokens/s |
| Accepted throughput | ~1320 tokens/s |
| Drafted throughput | ~1810 tokens/s |

---

## 已修复的问题

### Fix 1: float16 → bfloat16 权重位级重解释（接受率 1% → 60.8%）

**文件**: `atom/model_ops/linear.py` — `LinearBase.weight_loader_process()`

**根因**: Eagle3 draft 模型 checkpoint 中 `fc.weight`、`midlayer.*` 等权重为 float16，模型参数 dtype 为 bfloat16。加载时使用 `view()` 做位级重解释而非 `to()` 做数值转换。

float16 和 bfloat16 指数/尾数位宽不同（float16: 5/10, bfloat16: 8/7），同一二进制位模式代表完全不同的数值：

```
float16  0.007324  →  bits 0x1F80
同样 bits 作为 bfloat16  →  0.000000（非规格化数）
```

结果：fc.weight 及 midlayer 所有权重加载后数值近似为零，draft 模型等于随机猜测。

**修复**: float16 ↔ bfloat16 改为 `to()` 数值转换，fp8 等同族格式保持 `view()` 位重解释：

```python
if param.data.dtype != loaded_weight.dtype:
    if param.data.element_size() == loaded_weight.element_size():
        incompatible = {torch.float16, torch.bfloat16}
        if {param.data.dtype, loaded_weight.dtype} == incompatible:
            loaded_weight = loaded_weight.to(param.data.dtype)   # 数值转换
        else:
            param.data = param.data.view(loaded_weight.dtype)    # fp8 保持原逻辑
    else:
        loaded_weight = loaded_weight.to(param.data.dtype)
```

**附带修复**: `atom/config.py` — eagle3 方法默认启用 `use_aux_hidden_state = True`（当 draft 模型 config 无 `eagle_config` 字段时）。

**提交**: `465344d`

### Fix 2: Eagle3 draft 模型双重 RMSNorm（接受率 60.8% → 64.5%）

**文件**: `atom/models/eagle3_llama.py`, `atom/spec_decode/eagle.py`

**根因**: `Eagle3LlamaModel.forward()` 只返回 post-norm 的 hidden states：

```python
# 修复前
hidden_states = self.norm(hidden_states)
return hidden_states  # 只有 post-norm
```

在 `eagle.py` 的 propose() 循环中，后续 draft step（i > 0）将这个已经 norm 过的结果直接传入下一轮迭代：

```python
hidden_states = sample_hidden_states  # 已经被 norm 过
```

下一轮 decoder layer 再次做 `hidden_norm(hidden_states)`，导致**双重 RMSNorm**，信号被压缩。

vLLM 的做法是返回两个值 `(post_norm, pre_norm)`：
- `post_norm` → 给 `compute_logits` 算 logits
- `pre_norm` → 给下一轮 draft step 做输入

**影响**: 第 1 个 draft token（i=0）不受影响（用的是 target 模型的 aux hidden states），但第 2、3 个 draft token 质量下降。

**修复**:

`eagle3_llama.py` 返回两个值：
```python
hidden_states_prenorm = hidden_states
hidden_states = self.norm(hidden_states)
return hidden_states, hidden_states_prenorm
```

`eagle.py` 分离 logits 输入和下一步输入：
```python
if is_eagle3:
    ret_hidden_states, ret_hidden_prenorm = model_output
    # logits 用 ret_hidden_states (post-norm)
    # 下一步用 ret_hidden_prenorm (pre-norm)
```

**状态**: 已验证，接受率 60.8% → 64.5%。

### Fix 3: aux layer IDs 默认值对齐 vLLM（接受率 64.5% → 65.9%）

**文件**: `atom/models/deepseek_v2.py` — `get_eagle3_aux_hidden_state_layers()`

**根因**: ATOM 默认公式 `(1, N//2-1, N-4)` = `(1,29,57)`（kimi-k2.5-eagle3 config 中明确指定），而 vLLM 使用 `(2, N//2, N-3)` = `(2,30,58)`。

**修复**: 把 ATOM 默认公式改为 `(2, N//2, N-3)` 对齐 vLLM。

```python
def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)
```

**影响**: 接受率 64.5% → 65.9% (+1.4%)，全部接受比例 46.5% → 48.4%。说明 layer (2,30,58) 包含的中间表示对 draft 预测略好。

---

## 剩余差距分析（ATOM 65.9% vs vLLM 72.9%）

### 已排除的差异

| 可能原因 | 分析结论 |
|----------|----------|
| 采样策略不同 | ❌ 排除。两者都使用 `argmax` 贪心采样 |
| Residual/norm 计算顺序 | ❌ 排除。数学等价：两者最终都计算 `norm(original_hidden + attn_output + mlp_output)` |
| `norm_before_residual` 配置 | ❌ 排除。kimi-k2.5-eagle3 config 中无此字段，vLLM 走 `_norm_after_residual` 路径，与 ATOM 等价 |
| `norm_before_fc` 配置 | ❌ 排除。两者都为 False，不做 fc 前的 norm |
| fc / combine_hidden_states 逻辑 | ❌ 排除。两者都是 `fc(cat(aux_hidden_states))` |
| CUDA graph 处理 aux hidden states | ❌ 排除。两者都在 graph capture 时保存 aux buffer，replay 时正确截取 |
| Aux hidden states 捕获时机 | ❌ 排除。两者都在 layer 执行前捕获 `hidden_states + residual` |
| context_lens 包含 rejected tokens | ❌ 排除。测试减去 `num_reject_tokens` 后接受率反而下降（64.5% → 61.1%）。原因：i=0 prefill pass 会重写所有位置（包括 rejected）的 KV cache，所以后续 decode 读到的都是有效新数据 |

### 已逐项排查、确认与 vLLM 等价的差异

| 可能原因 | 分析结论 |
|----------|----------|
| Position `+1` 偏移 | ❌ 排除。ATOM i=0 用 `target_positions+1`，vLLM 用 `target_positions`，但 RoPE 是相对位置编码，self-attention 中常数偏移不影响 attention 输出。后续 step 两者增量也相同 |
| QKV 输入维度 / bias / MLP 配置 | ❌ 排除。`qkv_input_size = 2*hidden_size`、`qkv_bias=False`、`intermediate_size=12288` 完全一致 |
| Aux 顺序 | ❌ 排除。两者按 layer idx 升序 append，concat 顺序一致 |
| `slot_mapping` / `kv_indptr` 推导 | ❌ 排除。ATOM `kv_indices_generate_triton + kv_indptr += cu_seqlens_q` 与 vLLM 的 fused kernel 数学等价；MLA block_size=1 + block_ratio=16 与 AiterBackend block_size=16 在物理 slot 索引上兼容 |
| Forward 返回值 (post_norm, pre_norm) | ❌ 排除。Fix 2 已对齐 vLLM 的 `self.norm(hidden_states, residual)` 双返回 |

### 待进一步排查的方向

#### 1. AMD ROCm vs CUDA 数值精度

ATOM Eagle3 draft 用 `paged_attention_asm`（aiter），vLLM 在 CUDA 上用 Flash Attention。两者数学等价但数值实现不同。bf16 累加顺序、softmax 实现细节都可能造成 logits 微小偏移，进而影响 argmax。

#### 2. MXFP4 target × bf16 draft 精度交互

Target 模型使用 MXFP4 量化，draft 模型为 bf16。两个 stack 上 MXFP4 dequant 实现差异可能让 target 输出分布漂移略不同，从而 draft（未量化）匹配度不同。

vLLM 测试也使用了相同的 MXFP4 模型（接受率 72.9%），所以量化本身存在但量化实现差异可能贡献部分 gap。

#### 3. Eagle3 draft 未启用 `torch.compile`

ATOM 把 draft 模型 `compilation_config.level = NO_COMPILATION`（`eagle.py:62`），vLLM 用 `@support_torch_compile`。可能导致融合模式不同。

**影响评估**: 主要影响性能而非准确率，但融合后的 op 顺序变化可能在 bf16 下产生微小数值差异。

#### 4. Warmup 阶段 NaN 问题

之前 warmup 阶段曾观察到 NaN，虽然推理阶段正常。可能存在边界条件下的数值不稳定。

**建议下一步**：在相同 prompt 下打印 ATOM 和 vLLM 的 draft top-5 logits，对比是否系统性偏移；或对单层做 fp32 路径测试，看 acceptance 是否提升。

---

## 修复影响分析

### 接受率分布变化

| 分布 | Fix 1 后 | Fix 2 后 | Fix 3 后 | 总变化 |
|------|:-:|:-:|:-:|:-:|
| 0 accepted (全拒) | 36% | 17.9% | 17.3% | ↓ 显著 |
| 1 accepted | 29% | 17.1% | 15.6% | ↓ |
| 2 accepted | 20% | 18.5% | 18.7% | ≈ |
| 3 accepted (全中) | 15% | 46.5% | 48.4% | ↑ 显著 |

Fix 2 后全部接受的比例从 15% 跳到 46.5%；Fix 3 进一步提升到 48.4%。说明消除双重 norm + 选用更优 aux layer 组合后，第 2、3 个 draft token 的预测质量持续提升。

### GSM8K Accuracy 稳定

所有配置下 accuracy 保持 92.6%-93.8% 范围，说明 rejection 机制正确工作，不会引入错误 token。

---

## 下一步建议

| 优先级 | 行动 | 预期效果 |
|:-:|------|----------|
| P1 | 对比 ATOM vs vLLM 的 draft model logits 输出 | 在相同输入下，两者 top-k 是否系统性偏移；可定位是数值精度还是 attention kernel 差异 |
| P2 | 单层 fp32 测试 | 把 RMSNorm/MLP 强制走 fp32，看 acceptance 是否提升；若提升，说明 bf16 累加误差是主因 |
| P2 | 排查 warmup 阶段 NaN 问题 | 确认数值稳定性 |
| P3 | 对 Eagle3 draft 启用 `torch.compile` | 与 vLLM 一致，可能微调融合精度 |

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `atom/model_ops/linear.py` | Fix 1: dtype 转换修复 |
| `atom/config.py` | Fix 1: aux hidden state 默认值 |
| `atom/models/eagle3_llama.py` | Fix 2: forward 返回 pre-norm |
| `atom/spec_decode/eagle.py` | Fix 2: propose 使用 pre-norm 做下一步输入 |
| `atom/models/deepseek_v2.py` | Fix 3: aux layer IDs 默认值 `(2,N//2,N-3)` 对齐 vLLM |
| `atom/models/deepseek_v2.py:1891-1908` | Aux hidden states 捕获 |
| `atom/model_engine/model_runner.py:610-621` | Aux layer 初始化 |
| `vllm_kimi_k25_eagle3_setup.md` | vLLM 独立部署文档 |

## 相关提交

| Commit | 说明 |
|--------|------|
| `465344d` | Fix 1: float16→bfloat16 权重修复 + aux 默认值 |
| `dd239d8` | Eagle3 初始实现 |
| `d52011d` - `b319e91` | Eagle3 KV cache、CUDA graph、tensor size 修复 |
