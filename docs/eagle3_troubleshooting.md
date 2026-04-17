# Eagle3 Speculative Decoding — Troubleshooting Log

## Issue 1: Draft Model Weight Loading Dimension Mismatch

**Symptom**

```
RuntimeError: The size of tensor a (2304) must match the size of tensor b (1536) at non-singleton dimension 0
```

启动时加载 Eagle3 draft 模型权重失败，所有 rank 同时报错。

**Root Cause**

`EagleProposer.__init__` 中用 target 模型的 `atom_config`（包含 target 的 `hf_config`）构建 Eagle3 draft 模型：

```python
self.model = model_class(self.config)  # self.config 是 target 的 config
```

Target 模型 (Kimi-K2.5) 的 `intermediate_size=18432`，TP=8 时每卡分片大小为 2304。
Eagle3 draft 模型的 `intermediate_size=12288`，TP=8 时每卡分片大小为 1536。

模型参数按 2304 分配，但 checkpoint 权重是 1536，导致 copy 时维度不匹配。

**Fix** (`atom/spec_decode/eagle.py`)

为 Eagle3 创建 config 浅拷贝，替换 `hf_config` 为 draft 模型自己的配置：

```python
if self.speculative_config.method == "eagle3":
    draft_atom_config = copy.copy(atom_config)
    draft_atom_config.hf_config = draft_model_hf_config
    self.model = model_class(draft_atom_config)
```

---

## Issue 2: Draft Attention v_cache is None at Runtime

**Symptom**

服务启动成功，首次请求 prefill 通过后，在 decode 阶段 drafter propose 时崩溃：

```
TypeError: reshape_and_cache_with_pertoken_quant: value_cache needs to be <class 'torch.Tensor'> but got <class 'NoneType'>
```

错误发生在 `attention_mha.py:209` 的 `rope_cache` 方法中，`self.v_cache` 为 `None`。

**Root Cause**

`Eagle3LlamaModel` 被 `@support_torch_compile` 装饰。装饰器根据 `compilation_config.level` 决定是否启用 torch.compile：

```python
# decorators.py line 419-425
self.do_not_compile = atom_config.compilation_config.level in [
    CompilationLevel.NO_COMPILATION,
    CompilationLevel.DYNAMO_AS_IS,
]
```

Eagle3 draft 模型继承了 target 的 `compilation_config`（level=PIECEWISE=3），`do_not_compile=False`。即使命令行传了 `--enforce-eager`，该标志只影响 CUDA graph 捕获，不影响 `@support_torch_compile` 装饰器的编译行为。

torch.compile/Dynamo 在首次 forward 时编译 Eagle3 模型，生成的编译图直接捕获了 attention module 的属性引用。但 KV cache 绑定（`allocate_kv_cache` 中设置 `module.k_cache` / `module.v_cache`）发生在模型构建之后、首次请求之前。编译图中的属性访问路径与运行时绑定的属性之间产生了不一致，导致 `v_cache` 为 `None`。

**Fix** (`atom/spec_decode/eagle.py`)

为 Eagle3 draft 模型的 config 副本中禁用 torch.compile：

```python
draft_compilation = copy.copy(atom_config.compilation_config)
draft_compilation.level = CompilationLevel.NO_COMPILATION
draft_atom_config.compilation_config = draft_compilation
```

**Note**: 修改代码后必须清除编译缓存，否则会加载旧的编译结果导致静默错误：

```bash
rm -rf /root/.cache/atom/*
```

---

## Issue 3: Draft Attention v_cache is None — layer_num Mismatch in kv_cache_data

**Symptom**

禁用 torch.compile 后（Issue 2 的 fix），相同的 `v_cache is None` 错误仍然出现：

```
TypeError: reshape_and_cache_with_pertoken_quant: value_cache needs to be <class 'torch.Tensor'> but got <class 'NoneType'>
```

**Root Cause**

`attention_mha.py:125` 通过 `fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"]` 查找 KV cache。`kv_cache_data` 按照 `allocate_kv_cache` 遍历顺序编号：target MLA 层占 `layer_0` ~ `layer_60`，Eagle3 draft 层分配到 `layer_61`。

但 `Eagle3LlamaModel.__init__` 中用 `layer_num=config.num_hidden_layers`（draft config，值为 1），导致 draft attention impl 查找 `kv_cache_data["layer_1"]`——这是 target 模型的第二个 MLA 层，其 `v_cache=None`（MLA 用统一 kv_cache，不拆分 k/v）。

对比 MTP 的做法：MTP draft 层的 `layer_idx` 从 `config.num_hidden_layers`（target config）开始编号，确保与 `kv_cache_data` 索引一致。

**Fix** (`atom/spec_decode/eagle.py` + `atom/models/eagle3_llama.py`)

在 `eagle.py` 中设置 `draft_layer_offset` 为 target 模型的层数：

```python
draft_atom_config.draft_layer_offset = atom_config.hf_config.num_hidden_layers
```

在 `eagle3_llama.py` 中使用该 offset：

```python
layer_offset = getattr(atom_config, "draft_layer_offset", 0)
self.midlayer = Eagle3LlamaDecoderLayer(
    config=config, cache_config=cache_config,
    prefix="midlayer", layer_num=layer_offset,
)
```

---

## Issue 4: Eagle3 Draft Layer Missing k_scale/v_scale for FP8 KV Cache

**Symptom**

Issue 3 修复后，Eagle3 draft attention 找到了正确的 `kv_cache_data` 条目，k_cache/v_cache 不再是 None。但使用 `--kv_cache_dtype fp8` 时，`reshape_and_cache_with_pertoken_quant` 需要 `k_dequant_scales`/`v_dequant_scales`，而 Eagle3 draft 层的这些值为 None：

```
TypeError: reshape_and_cache_with_pertoken_quant: k_dequant_scales needs to be <class 'torch.Tensor'> but got <class 'NoneType'>
```

**Root Cause**

`allocate_kv_cache` 中绑定 k_scale/v_scale 时，条件 `not is_eagle3_draft` 跳过了 Eagle3 draft 层。只为 target 模型的非 MLA 层从 `self.kv_scale` 分配 scale，但 Eagle3 draft 层没有对应的 scale tensor。

**Fix** (`atom/model_engine/model_runner.py`)

1. 在 `eagle3_kv_cache` 分配后紧接着分配 `eagle3_kv_scale`：

```python
self.eagle3_kv_scale = torch.zeros(
    2, draft_hf.num_hidden_layers,
    self.num_physical_kvcache_blocks,
    eagle3_num_kv_heads, self.physical_block_size,
    dtype=dtypes.fp32, device="cuda",
)
```

2. 在绑定循环中，为 Eagle3 draft 层设置 scale：

```python
if config.kv_cache_dtype == "fp8":
    if is_eagle3_draft:
        module.k_scale = self.eagle3_kv_scale[0, eidx]
        module.v_scale = self.eagle3_kv_scale[1, eidx]
    else:
        module.k_scale = self.kv_scale[0, attn_idx]
        module.v_scale = self.kv_scale[1, attn_idx]
```

---

## Issue 5: Eagle3 MHA Decode Needs block_tables / context_lens

**Symptom**

```
AttributeError: 'NoneType' object has no attribute 'stride'
```

在 `attention_mha.py:412`，`attn_metadata.block_tables` 为 `None`。

**Root Cause**

Eagle3 draft 模型使用 MHA（标准 paged attention），decode 时需要 `block_tables` 和 `context_lens` 来索引 KV cache 的物理块。但 target 模型使用 MLA，只填充 `kv_indptr` / `kv_indices`（FlashInfer ragged 格式），不设置 `block_tables`。MTP draft 模型不存在此问题，因为它们也使用 MLA attention。

**Fix** (`atom/spec_decode/eagle.py`)

在 `propose` 方法的 `i==0` 迭代后（`context.is_prefill = False` 之后），从 `kv_indptr` / `kv_indices` 构建 `block_tables` 和 `context_lens`：

```python
if self.speculative_config.method == "eagle3":
    num_blocks_per_seq = kv_indptr[1:bs+1] - kv_indptr[:bs]
    max_num_blocks = int(num_blocks_per_seq.max().item())
    block_tables = torch.zeros(bs, max_num_blocks, dtype=torch.int32, device=...)
    for s in range(bs):
        nb = int(num_blocks_per_seq[s].item())
        st = int(kv_indptr[s].item())
        block_tables[s, :nb] = kv_indices[st:st+nb].to(torch.int32)
    attn_metadata.block_tables = block_tables
    attn_metadata.context_lens = (
        (num_blocks_per_seq - 1) * block_size + kv_last_page_lens[:bs]
    ).to(torch.int32)
```

每次 `prepare_mtp_decode` 更新 `kv_indptr` 后，也需要同步更新 `block_tables` 和 `context_lens`。

---

## Issue 6: ASM Paged Attention Kernel Has No Heuristic for Eagle3

**Symptom**

```
get_heuristic_kernel: cannot get heuristic kernel! q_type:bf16 kv_type:fp8 gqa:1 ... block_size:1
terminate called without an active exception
```

服务启动成功，首次请求 prefill 通过，decode 第一步 draft propose 时崩溃。

**Root Cause**

`attention_mha.py:533` 的 `dispatch_backend` 方法根据 `use_triton_attn` 选择 decode 后端。`use_triton_attn` 在 `__init__` 中设置：

```python
self.use_triton_attn = self.sliding_window != -1 or self.head_dim != 128
```

Eagle3 draft attention 的 `sliding_window` 默认为 `None`→`-1`，`head_dim=128`，因此 `use_triton_attn=False`，decode 路由到 `paged_attention_asm`。

ASM paged attention kernel 不支持 Eagle3 的参数组合（`gqa=1`，即 `num_heads == num_kv_heads`，且 `kv_type=fp8`），没有对应的 heuristic kernel，直接 abort。

**Fix** (`atom/models/eagle3_llama.py`)

在 `Eagle3LlamaAttention.__init__` 中，向 `Attention` 构造函数传入 `per_layer_sliding_window=0`：

```python
self.attn = Attention(
    ...,
    per_layer_sliding_window=0,
)
```

`sliding_window=0` 使 `use_triton_attn = True`（因为 `0 != -1`），decode 路由到 `paged_attention_triton`。Triton kernel 中 `sliding_window=0` 被视为无滑动窗口（`if self.sliding_window > 0:` 条件不满足）。

同时需修改 `attention_mha.py` 中 `prefill_attention` 和 `prefill_attention_triton` 的 sliding_window 条件，将 `self.sliding_window is not None` 改为 `self.sliding_window > 0`。否则 `sliding_window=0` 在 prefill 路径会传 `window_size=(0, 0, 0)` 给 `fmha_v3_varlen_fwd`，导致 `invalid argument` 错误。

---

## Issue 7: Triton PA Kernel `pa_decode_gluon` 不支持 block_size=1

**Symptom**

```
AssertionError: kv_block_size == 1 not in [16, 64, 1024]
```

在 `pa_decode_gluon.py:3445`。

**Root Cause**

MLA target 模型使用 `block_size=1`（`AiterMLAMetadataBuilder.block_size = 1`）。Eagle3 KV cache 继承了 `physical_block_size=1`，因此 k_cache 形状为 `[num_blocks, kv_heads, head_dim//x, 1, x]`。

`paged_attention_triton` 在 line 333 从 k_cache shape 读取 block_size：`num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape`，得到 `block_size=1`，传给 `pa_decode_gluon`。但该 triton kernel 只支持 `block_size in [16, 64, 1024]`。

**Fix** (`atom/model_ops/attention_mha.py`)

Issue 7 和 Issue 8 的根本原因相同：现有的 paged attention kernels 都不支持 block_size=1 的 decode：
- `pa_decode_gluon`：只支持 block_size in [16, 64, 1024]
- `unified_attention`（CK `fmha_v3_varlen_fwd` + block_table）：block_size=1 + fp8 时 GPU 非法内存访问
- `cp_mha_gather_cache` + `flash_attn_varlen_func`：triton gather kernel 在 block_size=1 时也产生 GPU 非法内存访问

解决方案：新增 `eagle3_decode_attention` 方法，采用纯 PyTorch gather + SDPA 策略，完全避开所有 native/triton kernel 的 block_size 限制：

```python
@mark_trace(prefix="eagle3_decode_attention", torch_compile=False)
def eagle3_decode_attention(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
    attn_metadata = fwd_ctx.attn_metadata
    block_tables = attn_metadata.block_tables
    bs = block_tables.shape[0]
    max_ctx = block_tables.shape[1]
    context_lens = attn_metadata.context_lens[:bs].clamp(max=max_ctx)

    num_blocks = k_cache.shape[0]
    # Reshape from fused_qk_rope_reshape_and_cache layout to [N, nh, hd]
    # K: [N, nh, hd//x, bs, x] -> permute(0,1,3,2,4) -> view(N, nh, hd)
    # V: [N, nh, hd, bs]        -> permute(0,1,3,2)   -> view(N, nh, hd)
    if k_cache.dim() == 5:
        k_flat = k_cache.permute(0,1,3,2,4).contiguous().view(num_blocks, nh, hd)
    if v_cache.dim() == 4:
        v_flat = v_cache.permute(0,1,3,2).contiguous().view(num_blocks, nh, hd)

    # PyTorch index gather (no kernel, no block_size constraint)
    gather_idx = block_tables.long()
    k_gathered = k_flat[gather_idx]  # [bs, max_ctx, nh, hd]
    v_gathered = v_flat[gather_idx]

    # fp8 dequant
    if self.kv_cache_dtype.startswith("fp8"):
        k_gathered = k_gathered.to(q.dtype) * k_scale
        v_gathered = v_gathered.to(q.dtype) * v_scale

    # Padding mask for variable-length sequences
    pad_mask = torch.arange(max_ctx) < context_lens.unsqueeze(1)

    # PyTorch native SDPA (no block_size constraint)
    o = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=pad_mask, scale=self.scale,
    )
    return o
```

修改 `dispatch_backend`：Eagle3 decode（`self.sliding_window == 0`）路由到 `eagle3_decode_attention`：

```python
if self.use_triton_attn:
    if self.sliding_window == 0:
        return self.eagle3_decode_attention
    return self.paged_attention_triton
```

优点：
- 完全绕过所有 paged attention kernel 的 block_size 限制
- 纯 PyTorch 操作，无 triton/CK kernel 兼容性问题
- Eagle3 只有 1 层 attention 且 decode 时 bs 较小，gather 开销可接受

迭代历史：
1. 第一版：`cp_mha_gather_cache` + `flash_attn_varlen_func` → GPU memory fault（triton gather kernel 不兼容 block_size=1）
2. 第二版：纯 PyTorch gather + SDPA → 成功避免 GPU crash

---

## Issue 8: CK Flash Attention `fmha_v3_varlen_fwd` + block_table 不支持 block_size=1

**Symptom**

```
Memory access fault by GPU node-X on address 0x...  Reason: Unknown.
```

所有 8 个 GPU 同时 crash，无 Python traceback。

**Root Cause**

Issue 7 的初始 fix 尝试将 Eagle3 decode 路由到 `prefill_attention_triton` → `unified_attention` → `fmha_v3_varlen_fwd`。该 CK kernel 在 block_size=1 + block_table 组合下产生非法内存访问（无论 fp8 还是 bf16）。这是 kernel 层面的限制，不是 Python 代码逻辑问题。

**Fix**

与 Issue 7 共用同一个 fix：`eagle3_decode_attention`（纯 PyTorch gather + SDPA）。见 Issue 7。

---

## Issue 9: `eagle3_decode_attention` 中 context_lens 与 block_tables 不同步

**Symptom**

```
RuntimeError: The expanded size of the tensor (29) must match the existing size (445)
    at dimension 3 in: [1, 8, 1, 29] vs [1, 1, 1, 445]
```

SDPA 的 attn_mask 形状不匹配。

**Root Cause**

`eagle3_decode_attention` 使用 `attn_metadata.context_lens` 构建 padding mask，但 `context_lens` 的值（445）远大于 `block_tables` 的列数（29）。

原因：`eagle.py:propose()` 中 `prepare_mtp_decode` 返回的 workinfos 通过 `attn_metadata.__dict__[k] = v` 写入 attn_metadata，虽然不直接覆盖 `context_lens`，但 `kv_indptr` 的更新可能导致 eagle3 的 `block_tables` 列数与 `context_lens` 不同步（因为 `context_lens` 是从 `(num_blocks_per_seq - 1) * block_size + kv_last_page_lens` 计算的原始 token 数，而 `block_tables` 只有 `max(num_blocks_per_seq)` 列）。

对于 block_size=1 这两个值应该相等，但在多次迭代后可能出现不同步。

**Fix** (`atom/model_ops/attention_mha.py`)

以 `block_tables.shape` 为权威来源：
```python
bs = block_tables.shape[0]
max_ctx = block_tables.shape[1]
context_lens = attn_metadata.context_lens[:bs].clamp(max=max_ctx)
```

---

## Issue 10: `eagle3_decode_attention` 在 propose i=0 多 token 场景下 q 形状不匹配

**Symptom**

```
RuntimeError: shape '[1, 8, 128]' is invalid for input of size 4096
```

在 `attention_mha.py:576`，`q.view(bs, self.num_heads, self.head_dim)` 报错。

**Root Cause**

Decode batch 中 spec decode 给每个 sequence 分配 `mtp_k + 1 = 4` 个 token（scheduler.py line 406: `num_new_tokens = self.mtp_k + 1`）。`propose_draft_token_ids` 将 `input_ids.gpu[1 : batch.total_tokens_num + 1]`（4 个 token）传给 `propose`。

在 propose 循环 i=0 时，draft 模型处理全部 4 个 token。此时 `context.is_prefill = False`（这是 decode batch），所以 `dispatch_backend` 路由到 `eagle3_decode_attention`。但该方法假设 `num_tokens == bs`（每个 sequence 1 个 token），用 `block_tables.shape[0]` (bs=1) reshape q，而 q 实际有 4 个 token（4 × 8 × 128 = 4096 元素），导致 reshape 失败。

本质上，propose i=0 是对 draft 模型的"prefill-like"多 token 前向传播，但 `context.is_prefill` 已经是 False。

**Fix** (`atom/model_ops/attention_mha.py`)

在 `eagle3_decode_attention` 入口检查 `num_tokens != bs`，用 SDPA 直接处理 fresh q/k/v（不依赖 block_table 和 cu_seqlens_k）：

```python
num_tokens = q.shape[0]
bs = block_tables.shape[0]
if num_tokens != bs:
    seq_len = num_tokens // bs
    q_sdpa = q.view(bs, seq_len, num_heads, head_dim).transpose(1, 2)
    k_sdpa = k.view(bs, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_sdpa = v.view(bs, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    # GQA expand ...
    o = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
    return o.transpose(1, 2).reshape(num_tokens, num_heads, head_dim)
```

后续 i=1, i=2 的单 token decode（num_tokens == bs）仍走 gather + SDPA 路径。

> **注意**：最初尝试回退到 `prefill_attention`（`flash_attn_varlen_func`），但 propose decode 阶段 `attn_metadata.cu_seqlens_k` 为 None，导致 `fmha_v3_varlen_fwd` 参数类型不匹配（见 Issue 11）。

---

## Issue 11: `prefill_attention` fallback 在 propose decode 阶段 cu_seqlens_k 为 None

**Symptom**

```
TypeError: fmha_v3_varlen_fwd(): incompatible function arguments.
```

`cu_seqlens_k` 参数为 None，不满足 `torch.Tensor` 类型要求。

**Root Cause**

Issue 10 的初始 fix 在多 token 时回退到 `prefill_attention`，但 `prefill_attention` 使用 `attn_metadata.cu_seqlens_k` 传给 `flash_attn_varlen_func`。在 propose decode 阶段（`context.is_prefill = False`），`attn_metadata.cu_seqlens_k` 未被设置（值为 None），因为 MLA decode metadata builder 不需要该字段。

**Fix** (`atom/model_ops/attention_mha.py`)

不再回退到 `prefill_attention`，改为直接在 `eagle3_decode_attention` 内用 SDPA 处理多 token 场景。将 fresh q/k/v reshape 为 `[bs, seq_len, heads, head_dim]`，用 `is_causal=True` 做 causal attention，完全不依赖 `cu_seqlens_k`、`flash_attn_varlen_func` 和 block_table。见 Issue 10 更新后的 fix。
