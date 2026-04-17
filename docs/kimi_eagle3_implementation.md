# Kimi-K2.5 Eagle3 Speculative Decoding — Implementation Document

## Overview

This document describes the implementation of Eagle3 speculative decoding support for the Kimi-K2.5 model in ATOM. Eagle3 is a speculative decoding method that uses an external lightweight draft model (Llama-based, single-layer, full-attention) to propose candidate tokens, guided by auxiliary hidden states collected from the target model's intermediate layers.

### Key Differences from Existing MTP Speculative Decoding

| 特性 | MTP | Eagle3 |
|------|-----|--------|
| Draft 模型来源 | Target 模型自带（权重嵌入在 target checkpoint） | 独立外部模型（单独 checkpoint 路径） |
| Draft 模型架构 | 与 target 同架构（MLA/GQA） | 独立 Llama full-attention 架构 |
| Hidden states 输入 | Target 最后一层 hidden states | Target 中间层 aux hidden states（3 层拼接投影） |
| Embedding / lm_head | 与 target 共享 | 独立（不共享） |
| KV Cache | 与 target 统一分配（MLA） | 独立分配（标准 MHA paged attention） |
| 权重加载 | `spec_decode=True`，需 key 重写 | `spec_decode=False`，权重 key 直接映射 |

### 启动命令

```bash
python -m atom.entrypoints.openai_server \
    --model /data/models/amd/Kimi-K2.5-MXFP4 \
    --trust-remote-code \
    --method eagle3 \
    --draft-model /data/models/kimi-k2.5-eagle3 \
    --eagle3-aux-layer-ids "1,29,57" \
    --num-speculative-tokens 3
```

---

## 1. Data Flow

```
Target Model Forward（61 layers）:
  layer[1]  hidden_states ──── aux[0]   [N, 7168]
  layer[29] hidden_states ──── aux[1]   [N, 7168]
  layer[57] hidden_states ──── aux[2]   [N, 7168]
  layer[60] hidden_states ──── target_hidden_states  [N, 7168]

ModelRunner.run_model():
  model_output = (hidden_states, [aux[0], aux[1], aux[2]])
  self._aux_hidden_states = [aux[0], aux[1], aux[2]]

EagleProposer.propose():
  ┌─ Step 0: 初始化 ──────────────────────────────────────────────────┐
  │ cat(aux[0], aux[1], aux[2], dim=-1)  → [N, 21504]               │
  │ self.model.combine_hidden_states()   → fc → [N, 7168]            │
  │ embed_tokens(input_ids)              → [N, 7168]                 │
  │ concat(normed_embed, normed_fc, dim=-1) → [N, 14336]             │
  │ → Eagle3LlamaAttention (64h/64kv, full-attn)                     │
  │ → MLP → norm → lm_head → argmax → draft_token[0]                │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ Step 1..k-1: 自回归 ────────────────────────────────────────────┐
  │ hidden_states = 上一步 draft 输出（不再用 aux）                     │
  │ embed_tokens(draft_token[i-1]) → embeds                          │
  │ concat(normed_embed, normed_hidden, dim=-1) → attention input     │
  │ → 同样的 decoder → argmax → draft_token[i]                       │
  └───────────────────────────────────────────────────────────────────┘
```

---

## 2. Modified Files

### 2.1 `atom/model_engine/arg_utils.py` — CLI Arguments

**Changes:**
- `--method` choices extended from `["mtp"]` to `["mtp", "eagle3"]`
- New argument `--draft-model`: path to external Eagle3 draft model checkpoint
- New argument `--eagle3-aux-layer-ids`: comma-separated target model layer indices for aux hidden state collection (e.g. `"1,29,57"`)
- `_get_engine_kwargs()`: when `method == "eagle3"`, constructs `SpeculativeConfig` with `model=self.draft_model`（independent path）instead of `model=self.model`（target path）

```python
# eagle3 path
SpeculativeConfig(
    method="eagle3",
    model=draft_model,           # /data/models/kimi-k2.5-eagle3
    num_speculative_tokens=...,
    eagle3_aux_layer_ids_str="1,29,57",
)

# mtp path (unchanged)
SpeculativeConfig(
    method="mtp",
    model=self.model,            # target model path
    num_speculative_tokens=...,
)
```

### 2.2 `atom/config.py` — SpeculativeConfig

**New fields:**
```python
use_aux_hidden_state: bool = False
eagle3_aux_layer_ids: list[int] = field(default_factory=list)
eagle3_aux_layer_ids_str: Optional[str] = None
```

**`__post_init__` logic:**
1. 加载 draft model 的 `hf_config`（`AutoConfig.from_pretrained(self.model)`）
2. 调用 `hf_config_override()` 做架构映射
3. 如果 `method == "eagle3"`：
   - 优先解析 CLI 字符串 `eagle3_aux_layer_ids_str` → `eagle3_aux_layer_ids`
   - 回退读取 draft model `config.json` 中的 `eagle_config`（lightseekorg 没有，NVIDIA 有）

**`hf_config_override()` — 新增 Eagle3 架构映射:**
```
LlamaForCausalLMEagle3 → Eagle3LlamaModel
```

这使得 `EagleProposer` 可以从 `support_eagle_model_arch_dict` 中找到正确的模型类。

### 2.3 `atom/models/eagle3_llama.py` — 新文件，Draft Model Implementation

#### Eagle3LlamaModel（顶层模型类）

```
Eagle3LlamaModel
├── embed_tokens     VocabParallelEmbedding [163840, 7168]  独立 embedding
├── fc               ReplicatedLinear [21504 → 7168]        3 aux 拼接投影
├── midlayer         Eagle3LlamaDecoderLayer                 单层 decoder
│   ├── input_layernorm    RMSNorm    normalize embedding
│   ├── hidden_norm        RMSNorm    normalize fc output
│   ├── self_attn          Eagle3LlamaAttention
│   │   ├── qkv_proj       QKVParallelLinear [14336 → 3×8192]  input = hidden×2
│   │   ├── o_proj         RowParallelLinear [8192 → 7168]
│   │   ├── rotary_emb     RoPE (theta=1M)
│   │   └── attn           Attention (MHA, 64h/64kv, sliding_window=0)
│   ├── mlp               Eagle3LlamaMLP
│   │   ├── gate_up_proj   MergedColumnParallelLinear [7168 → 2×12288]
│   │   └── down_proj      RowParallelLinear [12288 → 7168]
│   └── post_attention_layernorm  RMSNorm
├── norm             RMSNorm          最终 norm
└── lm_head          ParallelLMHead [7168 → 163840]   独立 lm_head
```

**关键设计:**

1. **`qkv_proj` input_size = hidden_size × 2 = 14336**：不是标准 Llama 的 `hidden_size=7168`。QKV 接收 `concat(normed_embed, normed_fc_output)` 作为输入。

2. **Dual-norm decoder**：`input_layernorm` 和 `hidden_norm` 分别对 embedding 和 fc output 做 RMSNorm，然后 concat 作为 attention 输入。

3. **`midlayer` 前缀**：模型属性直接命名为 `self.midlayer`，与 checkpoint 权重 key `midlayer.*` 完全匹配，无需 key 重写。

4. **`per_layer_sliding_window=0`**：强制 `use_triton_attn=True`，避免 ASM paged attention kernel 不支持 Eagle3 的 `gqa=1` + `fp8` 参数组合（见 Troubleshooting Issue 6）。

5. **`draft_layer_offset`**：attention `layer_num` 从 target 模型层数开始编号（e.g., 61），确保 `kv_cache_data["layer_61"]` 映射到正确的 Eagle3 KV cache 条目（见 Troubleshooting Issue 3）。

6. **`packed_modules_mapping`**：处理 Q/K/V 融合加载和 gate/up 融合加载：
   ```python
   packed_modules_mapping = {
       "q_proj": ("qkv_proj", "q"),
       "k_proj": ("qkv_proj", "k"),
       "v_proj": ("qkv_proj", "v"),
       "gate_proj": ("gate_up_proj", 0),
       "up_proj": ("gate_up_proj", 1),
   }
   ```

**权重 key → 模型属性映射:**

| Checkpoint Key | Model Attribute |
|---|---|
| `embed_tokens.weight` | `embed_tokens` |
| `fc.weight` | `fc` |
| `midlayer.hidden_norm.weight` | `midlayer.hidden_norm` |
| `midlayer.input_layernorm.weight` | `midlayer.input_layernorm` |
| `midlayer.self_attn.q_proj.weight` | `midlayer.self_attn.qkv_proj`（q 部分） |
| `midlayer.self_attn.k_proj.weight` | `midlayer.self_attn.qkv_proj`（k 部分） |
| `midlayer.self_attn.v_proj.weight` | `midlayer.self_attn.qkv_proj`（v 部分） |
| `midlayer.self_attn.o_proj.weight` | `midlayer.self_attn.o_proj` |
| `midlayer.mlp.gate_proj.weight` | `midlayer.mlp.gate_up_proj`（0 部分） |
| `midlayer.mlp.up_proj.weight` | `midlayer.mlp.gate_up_proj`（1 部分） |
| `midlayer.mlp.down_proj.weight` | `midlayer.mlp.down_proj` |
| `midlayer.post_attention_layernorm.weight` | `midlayer.post_attention_layernorm` |
| `norm.weight` | `norm` |
| `lm_head.weight` | `lm_head` |

### 2.4 `atom/spec_decode/eagle.py` — EagleProposer

#### 2.4.1 Architecture Registry

```python
support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
    "Eagle3LlamaModel": "atom.models.eagle3_llama.Eagle3LlamaModel",  # 新增
}
```

#### 2.4.2 `__init__` — Draft Model Construction

Eagle3 和 MTP 的 draft 模型构建方式不同：

```
MTP:   model_class(self.config)                    → 用 target 的 atom_config
Eagle3: model_class(draft_atom_config)             → 用修改后的 atom_config 副本
```

Eagle3 需要对 `atom_config` 做三处修改：
1. **`hf_config` → draft model config**：draft 模型有独立的 `intermediate_size`, `num_attention_heads` 等参数
2. **`compilation_config.level` → `NO_COMPILATION`**：禁用 torch.compile 避免 KV cache 绑定与 Dynamo 编译图不一致（见 Troubleshooting Issue 2）
3. **`draft_layer_offset` → target 层数**：确保 KV cache layer_num 映射正确（见 Troubleshooting Issue 3）

#### 2.4.3 `load_model` — Weight Loading

```
MTP 路径:
  load_model(self.model, self.config.model,       ..., spec_decode=True)
  → 从 target checkpoint 加载，需要 key 重写（spec_decode=True）
  → embed_tokens/lm_head 共享 target 模型

Eagle3 路径:
  load_model(self.model, self.speculative_config.model, ..., spec_decode=False)
  → 从 draft checkpoint 加载，key 直接映射（spec_decode=False）
  → embed_tokens/lm_head 独立，不共享
```

#### 2.4.4 `propose` — Aux Hidden States Processing

新增 `aux_hidden_states: Optional[list[torch.Tensor]]` 参数。

**第一次迭代（i=0）前的初始化：**
```python
if aux_hidden_states is not None:
    concat_aux = torch.cat(aux_hidden_states, dim=-1)   # [N, 21504]
    hidden_states = self.model.combine_hidden_states(concat_aux)  # [N, 7168]
else:
    hidden_states = target_hidden_states  # MTP 路径
```

**i=0 多 token 场景**：decode batch 中 scheduler 分配 `mtp_k + 1` 个 token 给每个 sequence。draft 模型处理全部 token 作为 "prefill-like" 前向传播。i=1 起每次处理单 token 自回归 decode。

**Eagle3 特有的 attn metadata 处理**：MLA target 模型使用 `kv_indptr/kv_indices`（FlashInfer ragged 格式），但 MHA draft 模型需要 `block_tables/context_lens`。在 `propose()` 中从 `kv_indptr/kv_indices` 构建 `block_tables` 和 `context_lens`，每次 `prepare_mtp_decode` 更新后同步更新。

### 2.5 `atom/models/deepseek_v2.py` — Aux Hidden States Collection

#### DeepseekV2Model

新增 `aux_hidden_state_layers: tuple[int, ...]` 属性。

Forward 循环中增加收集逻辑：
```python
aux_hidden_states = []
for idx in range(self.start_layer, self.end_layer):
    layer = self.layers[idx]
    if idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(
            hidden_states if residual is None else hidden_states + residual
        )
    hidden_states, residual = layer(positions, hidden_states, residual)

hidden_states, _ = self.norm(hidden_states, residual)
if aux_hidden_states:
    return hidden_states, aux_hidden_states   # tuple 返回
return hidden_states
```

收集的是 **pre-layer 值**（`hidden_states + residual`），即该层的输入，与 `llama.py` 和 `gpt_oss.py` 保持一致。

#### DeepseekV2ForCausalLM

新增两个方法：
- `set_aux_hidden_state_layers(layers)`: 设置需要收集 aux hidden states 的层索引
- `get_eagle3_aux_hidden_state_layers()`: 返回默认 aux 层索引 `(1, num_layers//2 - 1, num_layers - 4)`

### 2.6 `atom/models/kimi_k25.py` — Passthrough

`KimiK25ForCausalLM` 是 `DeepseekV2ForCausalLM` 的包装类，透传两个 Eagle3 方法：
```python
def set_aux_hidden_state_layers(self, layers):
    self.language_model.set_aux_hidden_state_layers(layers)

def get_eagle3_aux_hidden_state_layers(self):
    return self.language_model.get_eagle3_aux_hidden_state_layers()
```

### 2.7 `atom/model_engine/model_runner.py` — Runtime Integration

#### 2.7.1 初始化

```python
self.eagle3_mode = (
    self.config.speculative_config is not None
    and self.config.speculative_config.method == "eagle3"
)
self._aux_hidden_states = None
```

模型加载完成后，设置 aux layer ids：
```python
if self.eagle3_mode and self.config.speculative_config.use_aux_hidden_state:
    aux_ids = self.config.speculative_config.eagle3_aux_layer_ids
    if not aux_ids and hasattr(self.model, "get_eagle3_aux_hidden_state_layers"):
        aux_ids = list(self.model.get_eagle3_aux_hidden_state_layers())
    if aux_ids:
        self.model.set_aux_hidden_state_layers(tuple(aux_ids))
```

#### 2.7.2 KV Cache — 分离分配

Eagle3 draft 模型使用标准 MHA（而非 MLA），其 KV cache 布局与 target 模型不同，必须独立分配。

**`_get_total_num_layers()`**：Eagle3 模式不把 draft 层计入 total layers（避免 MLA cache 多分配无用层）。

**`_compute_block_bytes()`**：Eagle3 draft 层的 block 字节数单独计算，添加到 total block_bytes 中。

**`allocate_kv_cache()`**：

```
Target MLA KV Cache:    [2, 61, blocks, block_size, kv_lora_rank + qk_rope_dim]
Eagle3 MHA KV Cache:    [2, 1,  blocks, block_size, num_kv_heads, head_dim]
Eagle3 MHA KV Scale:    [2, 1,  blocks, num_kv_heads, block_size]  (fp8 only)
```

- Eagle3 KV cache 使用与 target 相同的 `num_physical_kvcache_blocks` 和 `physical_block_size`
- KV scale tensor 用于 fp8 per-token quantization 的 dequant

**KV Cache binding**：`allocate_kv_cache` 的 binding 循环中，通过 `model_name == "draft"` 和 `eagle3_mode` 判断是否为 Eagle3 draft 层，使用 `eagle3_kv_cache` 和 `eagle3_kv_scale` 替代 `kv_cache` 和 `kv_scale`。使用独立的 `eagle3_draft_layer_id` 计数器索引。

#### 2.7.3 `run_model` — 解包 Tuple 输出

```python
model_output = self.model(input_ids, positions)
if isinstance(model_output, tuple):
    hidden_states, self._aux_hidden_states = model_output
else:
    hidden_states = model_output
    self._aux_hidden_states = None
```

#### 2.7.4 `propose_draft_token_ids` — 传递 Aux

```python
draft_tokens = self.drafter.propose(
    ...,
    aux_hidden_states=self._aux_hidden_states,   # 新增
)
```

#### 2.7.5 CUDA Graph — 暂时禁用

Eagle3 模式跳过 CUDA graph capture：
```python
if self.eagle3_mode:
    self.graphs = {}
    self.graph_logits = {}
    self.graph_bs = [self.config.max_num_seqs]
    return
```

原因：
- `_aux_hidden_states` 是动态 list，不兼容 CUDA Graph 的静态张量约束
- Target 模型返回 tuple（含 aux list），graph replay 无法处理
- 后续优化可预分配固定 shape buffer `[3, N, 7168]`，replay 时 `copy_` 填入

同时在 `run_model` 中，`eagle3_mode` 强制走 eager path（不进入 graph replay 分支）。

### 2.8 `atom/model_ops/attention_mha.py` — Eagle3 Decode Attention

#### 2.8.1 `sliding_window` 条件修复

将 `prefill_attention` 和 `prefill_attention_triton` 中的 sliding_window 判断从 `self.sliding_window is not None` 改为 `self.sliding_window > 0`。原因：Eagle3 使用 `sliding_window=0` 来触发 `use_triton_attn=True`，但 `0` 不应被视为有效的滑动窗口大小。

#### 2.8.2 `eagle3_decode_attention` — 新增方法

**为什么需要新方法？**

Target 模型使用 MLA + `block_size=1`。Eagle3 draft 模型继承了 `block_size=1`，但现有的所有 paged attention kernels 都不支持 `block_size=1` 的 decode：
- `pa_decode_gluon`（Triton）: 只支持 block_size ∈ {16, 64, 1024}
- `fmha_v3_varlen_fwd`（CK）+ block_table: block_size=1 导致 GPU memory fault
- `cp_mha_gather_cache`（Triton gather）: block_size=1 导致 GPU memory fault

**实现：纯 PyTorch gather + SDPA**

```
Eagle3 Decode Attention 分两个路径：

Path A: 多 token (num_tokens != bs，i=0 的 propose)
  → q/k/v 都是 fresh（本次 forward 产生）
  → reshape 为 [bs, seq_len, heads, head_dim]
  → torch.nn.functional.scaled_dot_product_attention(is_causal=True)
  → 不依赖 KV cache 和 block_table

Path B: 单 token decode (num_tokens == bs, i≥1 的 propose)
  → 新 k/v 已写入 cache（rope_cache 步骤）
  → 从 k_cache/v_cache 按 block_tables gather 历史 KV
  → k_cache [num_blocks, nh, hd//x, 1, x] → permute → [num_blocks, nh, hd]
  → v_cache [num_blocks, nh, hd, 1] → permute → [num_blocks, nh, hd]
  → block_tables.long() 做 index gather → [bs, max_ctx, nh, hd]
  → fp8 dequant if needed
  → padding mask (arange < context_lens)
  → SDPA with attn_mask
```

**性能考量**：Eagle3 只有 1 层 attention 且 draft decode 的 bs 较小，gather 开销可接受。纯 PyTorch 操作避免了所有 native/triton kernel 的 block_size 限制。

#### 2.8.3 `dispatch_backend` 路由修改

```python
if self.use_triton_attn:
    if self.sliding_window == 0:         # Eagle3 draft attention
        return self.eagle3_decode_attention
    return self.paged_attention_triton   # 其他非 MLA attention
```

`sliding_window=0` 是 Eagle3 draft attention 的标识：
- `0 != -1` → `use_triton_attn = True`（进入 triton 分支）
- `0 == 0` → 路由到 `eagle3_decode_attention`（而非 triton PA kernel）
- `0 > 0` 为 False → 所有 sliding window 相关逻辑跳过（视为无滑动窗口）

---

## 3. Files Not Modified

| File | Reason |
|------|--------|
| `atom/model_loader/loader.py` | Eagle3 调用路径传 `spec_decode=False`，绕过 key 重写逻辑 |
| `atom/model_ops/linear.py` | 标准 linear 层无需修改 |
| `atom/model_ops/layernorm.py` | 标准 RMSNorm 无需修改 |

---

## 4. Troubleshooting Summary

实现过程中遇到并解决了 11 个问题，按类别整理：

### 4.1 模型构建阶段

| Issue | 问题 | 根因 | Fix |
|-------|------|------|-----|
| #1 | Weight shape mismatch (2304 vs 1536) | Draft 模型用 target 的 `hf_config` 构建，`intermediate_size` 不匹配 | 为 Eagle3 创建 config 浅拷贝，替换 `hf_config` 为 draft model 自己的配置 |
| #2 | v_cache is None（torch.compile） | `@support_torch_compile` 编译图捕获了 None 的 cache 属性引用 | 禁用 draft model 的 torch.compile（`level=NO_COMPILATION`） |
| #3 | v_cache is None（layer_num 错位） | Draft attention 查找 `kv_cache_data["layer_1"]` 而非 `["layer_61"]` | 设置 `draft_layer_offset = target.num_hidden_layers` |
| #4 | k_scale/v_scale is None（fp8） | `allocate_kv_cache` 跳过了 Eagle3 draft 层的 scale 分配 | 分配独立的 `eagle3_kv_scale` 并在 binding 循环中绑定 |

### 4.2 Decode Attention 阶段

| Issue | 问题 | 根因 | Fix |
|-------|------|------|-----|
| #5 | block_tables is None | MLA target 只设 `kv_indptr`/`kv_indices`，不设 `block_tables` | 在 propose() 中从 kv_indptr/kv_indices 构建 block_tables |
| #6 | ASM PA kernel abort (gqa=1) | ASM kernel 无 `gqa=1 + fp8` heuristic | 设 `per_layer_sliding_window=0` 强制 triton 路径 |
| #7 | Triton PA kernel assert (block_size=1) | `pa_decode_gluon` 只支持 block_size ∈ {16, 64, 1024} | 新增 `eagle3_decode_attention`（PyTorch gather + SDPA） |
| #8 | CK kernel GPU memory fault | `fmha_v3_varlen_fwd` + block_table 不兼容 block_size=1 | 同 Issue 7 |
| #9 | context_lens vs block_tables 不同步 | SDPA attn_mask 形状不匹配 | 以 `block_tables.shape` 为权威，`context_lens.clamp(max=max_ctx)` |
| #10 | q reshape 失败（多 token） | i=0 多 token propose 被路由到单 token decode path | 检测 `num_tokens != bs` 走 SDPA causal attention |
| #11 | cu_seqlens_k is None | propose decode 阶段 MLA metadata 不设 cu_seqlens_k | 不回退 prefill_attention，在 eagle3_decode_attention 内用 SDPA 处理 |

### 4.3 问题链依赖关系

```
Issue 1 (config mismatch)
  → Issue 2 (torch.compile)
    → Issue 3 (layer_num offset)
      → Issue 4 (fp8 scale)
        → Issue 5 (block_tables)
          → Issue 6 (ASM kernel)
            → Issue 7 (Triton PA block_size)
            → Issue 8 (CK kernel block_size)
              → Issue 9 (context_lens sync)
              → Issue 10 (multi-token reshape)
                → Issue 11 (cu_seqlens_k fallback)
```

每个 issue 都是在前一个 issue 修复后暴露的。核心挑战是 **Eagle3 MHA draft 运行在 MLA target 模型的基础设施上**，两者在 KV cache 布局、attention metadata 格式、paged attention kernel 选择上存在根本差异。

---

## 5. Known Limitations & Future Work

### 5.1 CUDA Graph 未启用

当前 Eagle3 模式跳过 CUDA graph capture。影响 decode 性能（每次 forward 都有 kernel launch overhead）。

**优化方向**：预分配固定 shape buffer `[3, max_bs, 7168]`，graph capture 时作为 placeholder，replay 时 `copy_` 填入 aux hidden states。需要同时处理 target model forward 返回 tuple 的问题。

### 5.2 Pipeline Parallel (PP) 未验证

PP 场景下 `start_layer/end_layer` 的切割可能导致 aux layers 不在当前 rank。`idx in self.aux_hidden_state_layers` 使用全局索引，需要确认与 PP 分片策略一致。

### 5.3 `eagle3_decode_attention` 性能

纯 PyTorch gather + SDPA 的性能不如 native kernel。但 Eagle3 只有 1 层 attention 且 decode 时 context length 有限，瓶颈更可能在 target model forward 而非 draft attention。

### 5.4 `block_tables` 构建使用 Python 循环

`eagle.py:propose()` 中构建和更新 `block_tables` 使用了 Python for 循环。在高并发（大 bs）场景可能成为瓶颈。可以用 `torch.scatter` 或自定义 CUDA kernel 替换。

### 5.5 不支持的模型组合

当前实现仅支持 **Kimi-K2.5（MLA target）+ lightseekorg Eagle3（Llama MHA draft）** 组合。以下组合不在本阶段范围：
- NVIDIA Kimi-K2.5-Thinking-Eagle3（MLA draft）
- GPT-OSS-120B + Eagle3（不同权重前缀）
- 其他 target 架构
