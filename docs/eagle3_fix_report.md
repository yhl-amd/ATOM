# Eagle3 投机解码低接受率问题修复报告

## 问题描述

使用 ATOM 引擎运行 Eagle3 投机解码（draft 模型：`kimi-k2.5-eagle3`，target 模型：`Kimi-K2.5-MXFP4`），接受率仅约 **1%**，远低于预期。Draft 模型几乎无法正确预测任何 token，投机解码完全失效。

## 根因分析

### 根本原因：float16 ↔ bfloat16 权重加载时位级重解释导致数值损坏

Eagle3 draft 模型 checkpoint（`kimi-k2.5-eagle3`）中，除 `embed_tokens.weight` 为 bfloat16 外，其余所有权重均为 **float16**：

| 权重 | checkpoint dtype | 模型参数 dtype |
|------|-----------------|---------------|
| embed_tokens.weight | bfloat16 | bfloat16 |
| fc.weight | **float16** | bfloat16 |
| midlayer.*.weight | **float16** | bfloat16 |
| norm.weight | **float16** | float32 |
| lm_head.weight | **float16** | bfloat16 |

ATOM 的 `LinearBase.weight_loader_process`（`atom/model_ops/linear.py`）在加载权重时，检测到 checkpoint dtype 与参数 dtype 不同但字节宽度相同（均为 2 字节），使用 `view()` 做**位级重解释**而非**数值转换**：

```python
# 修复前的代码
if (
    param.data.dtype != loaded_weight.dtype
    and param.data.element_size() == loaded_weight.element_size()
):
    param.data = param.data.view(loaded_weight.dtype)  # 位重解释，不转换数值
```

float16 和 bfloat16 的指数/尾数位宽不同（float16: 5/10，bfloat16: 8/7），同一二进制位模式在两种格式下代表完全不同的数值：

```
float16  0.007324 → bits 0x1F80
同样的 bits 作为 bfloat16 → 0.000000（非规格化数，近似为零）
```

结果：**fc.weight 及 midlayer 所有权重加载后数值近似为零**，draft 模型等于在随机猜测。

### 次要问题：use_aux_hidden_state 默认值缺失

`kimi-k2.5-eagle3` 的 config.json 中没有 `eagle_config` 字段。`atom/config.py` 在解析时，找不到 `eagle_config` 就不设置 `use_aux_hidden_state`，导致其保持默认值 `False`。虽然可以通过命令行 `--eagle3-aux-layer-ids` 绕过，但缺少兜底逻辑不够健壮。

## 修复内容

### 1. `atom/model_ops/linear.py` — 权重加载 dtype 转换

float16 ↔ bfloat16 改为 `to()` 做数值转换；fp8 等同族格式仍保持 `view()` 位重解释：

```python
# 修复后
if param.data.dtype != loaded_weight.dtype:
    if param.data.element_size() == loaded_weight.element_size():
        incompatible = {torch.float16, torch.bfloat16}
        if {param.data.dtype, loaded_weight.dtype} == incompatible:
            loaded_weight = loaded_weight.to(param.data.dtype)   # 数值转换
        else:
            param.data = param.data.view(loaded_weight.dtype)    # fp8 等保持原逻辑
    else:
        loaded_weight = loaded_weight.to(param.data.dtype)       # 不同字节宽度也做转换
```

### 2. `atom/config.py` — eagle3 默认启用 aux hidden states

当 method 为 eagle3 但 draft 模型 config 中无 `eagle_config` 时，默认 `use_aux_hidden_state = True`：

```python
if eagle_cfg:
    self.use_aux_hidden_state = eagle_cfg.get("use_aux_hidden_state", False)
    ...
else:
    self.use_aux_hidden_state = True  # 新增兜底
```

## 修复效果

### 测试配置

- Target 模型：`Kimi-K2.5-MXFP4`（MXFP4 量化）
- Draft 模型：`kimi-k2.5-eagle3`
- TP=8，`--num-speculative-tokens 3`，`--eagle3-aux-layer-ids "1,29,57"`
- 硬件：AMD ROCm

### 修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| fc.weight abs_mean | 0.000000 | 0.007324 |
| fc 层输出 abs_mean | 0.000000 | 2.34 |
| 接受率（Acceptance Rate） | ~1% | **~37%** |
| 平均 tokens/forward | ~1.01 | **~2.14** |
| 接受分布 (0/1/2/3 accepted) | 99%/1%/0%/0% | 36%/29%/20%/15% |

### 与官方数据对比

官方 benchmark 数据来自 [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) README，配置为 `topk=1, num_steps=3, num_draft_tokens=4`。

| 数据集 | 官方 accept_length | ATOM accept_length |
|--------|-------------------|-------------------|
| MTBench | 2.687 | — |
| GSM8K | 3.201 | — |
| HumanEval | 3.285 | — |
| MATH500 | 3.342 | — |
| 综合（长文本生成） | ~2.7–3.3 | **~2.14** |

ATOM 的 accept_length 与官方仍有差距，可能的原因：

1. **Target 模型量化差异**：ATOM 使用 MXFP4 量化版本，官方示例命令使用的是未量化的 `moonshotai/Kimi-K2.5`（`--dtype bfloat16`）。量化会导致 target 模型输出分布偏移，降低 draft 预测准确性。但官方跑 benchmark 时的具体配置 README 中未明确说明。
2. **Attention backend 兼容性**：Eagle3 draft 使用标准 Llama 全注意力（`AiterBackend`），target 使用 MLA（`AiterMLABackend`）。两者共享 attn_metadata，格式兼容性可能存在问题（warmup 阶段观察到 NaN）。
3. **Aux layer IDs**：当前使用 `[1, 29, 57]`，官方可能使用不同的层组合。

### 建议后续验证

1. 使用未量化的 Kimi-K2.5 模型重跑 benchmark，确认量化对接受率的影响
2. 排查 warmup 阶段 NaN 问题，确认 attention metadata 在 Eagle3 draft 模型中的正确性
3. 在官方 benchmark 脚本（SpecForge `bench_eagle3.py`）上跑标准数据集，与官方数据直接对比
