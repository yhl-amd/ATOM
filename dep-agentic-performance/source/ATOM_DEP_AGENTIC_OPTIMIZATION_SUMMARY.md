# ATOM DPA + EP 原生 RCCL 长上下文 Agentic Decode 优化总结

更新时间：2026-09-01

> 当前提交为 [ROCm/ATOM#2110](https://github.com/ROCm/ATOM/pull/2110)：只包含 native RCCL / EPLB-off 路径。本文保留 Decode EPLB 的历史实验数据用于解释取舍，但相关配置、采样、重排与 freeze 实现已从新 PR 删除，不代表当前代码能力。

## 1. 摘要

本轮工作的目标是在不依赖 MoRI 的情况下，为 ATOM 补齐可运行的 DPA8 + EP8 路径，并针对 DeepSeek-V4-Pro 的长上下文 AgentX Decode 场景进行优化和验证。

最终形成的方案为：

```text
DPA8
+ EP8
+ ATOM native RCCL hybrid MoE transport
+ EPLB off
+ LOCAL_REPLICA shared-expert global-row round-robin（fusion 生效时）
+ DP session affinity
```

最重要的结果如下：

- 原生 RCCL 路径已经可以在 8×MI355X 上稳定运行 DPA8 + EP8，不再要求安装 MoRI。
- `ATOM_DP_SESSION_AFFINITY=1` 是 Agentic 场景收益最大的优化：在同一套 c48 配置下，总 token 吞吐提高 84.98%，P90 Interactivity 提高 265.9%。
- 完整 3600 秒测试中，c48、c64、c96、c128 四档均超过最新官方 ATOM DPA CI 的同并发点。
- c64 是交互性与吞吐较好的 Pareto 点；c96 是更偏吞吐的折中点；c128 吞吐最高，但边际收益已下降。
- 同环境严格 A/B 中，关闭 EPLB 的完成请求数、总吞吐与尾延迟更好；因此新 PR 不再包含 Decode EPLB prototype。
- 与 B200 vLLM/SGLang CI 相比，MI355X DEP 仍有明显差距，重点瓶颈仍是 MoE 通信融合、低精度传输、TBO/计算通信重叠和高并发调度，而不是简单继续增加 concurrency。

> 说明：本文中的最终成绩来自完整 warmup + 3600 秒 AgentX profile。文中出现的 120 秒结果只用于工程诊断，不作为正式榜单结论。

## 2. 测试范围与指标口径

### 2.1 历史 3600 秒测试配置

```text
模型：DeepSeek-V4-Pro
硬件：8×AMD MI355X，单节点
场景：InferenceX AgentX MVP
数据集：semianalysis_cc_traces_weka_062126
Trace 数：393
随机种子：42
权重：FP4
KV Cache：FP8
Index Cache：FP4
Speculative decoding：MTP-3
并行：DPA8 + EP8
MoE 通信：--all2all-backend rccl
EPLB：decode-aware，单次重排后冻结（历史实验分支；新 PR 默认关闭）
Session affinity：开启
TBO：关闭
每档测试：完整 warmup + 3600 秒 profile
```

每个 concurrency 都重新启动服务，避免 KV cache、EPLB placement 或进程状态在不同测试之间串扰。这组 3600 秒结果用于记录研发过程，不作为新 PR 的 EPLB-off 单变量性能结论。

### 2.2 榜单指标

本文重点使用两个指标：

```text
P90 Interactivity
= 1000 / P90 Full-Response ITL(ms)

Output Token Throughput per Chip
= 全机 output_token_throughput / 8
```

两者都是越高越好。`Total token throughput` 也会列出，但 AgentX 长上下文中大量输入 token 来自 prefix-cache read，因此它不能单独代表 Decode 计算速度。

## 3. 原始问题

### 3.1 ATOM 的 DPA 不等于 DPA + EP

ATOM 中：

```text
-tp 8 --enable-dp-attention
```

会把原来的 TP8 改写为 TP1 × DP8，使每张 GPU 独立处理 Attention 和 KV Cache。但只有继续加入：

```text
--enable-expert-parallel
```

MoE 才会形成 EP8。也就是说：

```text
Attention：DP8 / 每个 rank TP1
MoE：EP8
总 GPU 数：8
```

### 3.2 没有 MoRI 时的旧 fallback

原始非 MoRI 路径在每个 MoE 层执行：

```text
AllGather(hidden states + router logits)
  -> 每张卡计算本地专家
  -> ReduceScatter(outputs)
```

它能保证正确性，但存在以下问题：

- 每个 rank 都接收整个通信组的 token，而不是只接收本地专家需要的数据。
- 每层包含全组同步，最慢 rank 会形成长尾。
- router logits、padding 和临时 tensor 增加显存带宽及 kernel launch 开销。
- 与 TBO 的两个 ubatch 并行执行时，collective 顺序可能不一致并导致 hang。

### 3.3 Decode EPLB 的实验动机（未进入新 PR）

原始 ATOM EPLB 可以让 Decode 使用已有的 expert placement，但负载统计与 rebalance 时钟只由 Prefill 推进。对于纯 Decode 节点或 Decode-heavy Agentic workload，它不能根据实际生成阶段的专家热点自动学习。

## 4. 完成的优化

### 4.1 原生 RCCL hybrid MoE transport

新增 `--all2all-backend rccl`，并保留以下选择：

```text
auto / mori：存在 MoRI 时使用 MoRI
rccl：       使用 ATOM 原生 RCCL hybrid backend
none：       强制使用原始 AllGather/ReduceScatter fallback
```

最终 RCCL backend 不是简单的 Python 动态 All-to-All，而是按 workload 分流：

```text
Uniform Decode
  -> 将 hidden、Top-K expert IDs、weights 打成一次紧凑 payload
  -> graph-safe AllGather
  -> 本地 AITER fused MoE
  -> ReduceScatter
  -> 支持 Decode CUDAGraph

Prefill / Mixed Batch，且 DP/EP 等宽
  -> 复用 scheduler 已同步的每-rank token counts
  -> variable-size AllGatherV
  -> 本地 AITER fused MoE
  -> ReduceScatterV

其他 topology
  -> 保留 variable-split routed all_to_all_single correctness path
```

相比第一版动态 routed All-to-All，这个 hybrid 设计消除了 Decode 每层的 count exchange、GPU→CPU split-size 同步和动态 tensor 分配，同时保留 Prefill/Mixed Batch 的正确性。

关键实现：

- [`ATOM/atom/model_ops/fused_moe/rccl_prepare_finalize.py`](ATOM/atom/model_ops/fused_moe/rccl_prepare_finalize.py)
- [`ATOM/atom/model_ops/fused_moe/routed_all2all.py`](ATOM/atom/model_ops/fused_moe/routed_all2all.py)
- [`ATOM/atom/model_ops/moe.py`](ATOM/atom/model_ops/moe.py)
- [`ATOM/atom/config.py`](ATOM/atom/config.py)

### 4.2 修复真实 V4 + MTP 路径的正确性问题

在真实 DeepSeek-V4-Pro 测试中发现并修复了几类 dummy-weight smoke 无法覆盖的问题：

- DPA gather 后 router 行数已变成全局大小，而 hash router 的 `input_ids` 仍是本地大小，导致 shape 不匹配。
- AITER EP sorting kernel 不能接收为非本 rank expert 写入的 `-1` ID；现在保留合法全局 expert ID，再由 expert mask 过滤。
- RCCL 返回的是精确大小 tensor，不能沿用 MoRI 固定 arena 的 `num_local_tokens` 语义。
- 无接收 token 的 rank 使用合法 dummy expert ID 和零权重，保证 fused-MoE launch shape 有效且结果为零。

关键实现：

- [`ATOM/atom/models/deepseek_v4.py`](ATOM/atom/models/deepseek_v4.py)
- [`ATOM/atom/model_ops/topK.py`](ATOM/atom/model_ops/topK.py)

### 4.3 Shared expert 负载均衡

V4 的 replicated shared expert 原本容易按请求来源 rank 聚集。AgentX 的请求长度高度不均，一个超长会话可能让某个 rank 的 shared-expert GEMM 长时间成为瓶颈。

Shared expert 实际有三条路径，不是在所有配置下都使用 round-robin：

- **Shared fusion 未生效**：shared expert 作为独立 FFN 运行，既不进入 EPLB，也不进入 RCCL shared round-robin。显式关闭 fusion，或 shared/routed expert 的量化规格不兼容，都可能走这条路径。
- **Shared fusion 生效 + EPLB 关闭，`LOCAL_REPLICA`**：每个 rank 保留一份固定 shared copy。RCCL AllGather/AllGatherV 后，gathered token 的 shared-expert owner 按 global token row 在 EP ranks 间 round-robin 分配，再通过 combine 汇总。
- **Shared fusion 生效 + EPLB 开启，`EPLB_ROUTED`**：shared expert 作为普通 logical expert 追加到 routed experts 之后，与它们一起参与负载统计、logical-to-physical mapping、replica placement 和权重迁移。这种布局下 physical slot 会随 rebalance 变化，因此不使用假设固定 shared slot 的 round-robin 改写。

当前 EPLB 的 replica dispatch 是 local-first：本 rank 有副本时优先用本地副本，否则才在远端副本间按 token 分散。因此，EPLB 会管理 shared expert，但对“每个 rank 都有 shared replica，而 source token 本身不均”的情况，它不等价于 global-row round-robin 的逐步严格均衡。

需特别区分“代码支持”和“实测已启用”。最终 c48–c128 V4-Pro 日志中模型声明 `n_routed_experts=384` 和 `n_shared_experts=1`，而 EPLB 初始化为 `num_logical=384, num_physical=384`。因此这批最终成绩中 EPLB 只管理 384 个 routed experts，shared expert 仍是 standalone 路径；不能把成绩归因于 `LOCAL_REPLICA` round-robin 或 shared-in-EPLB。

以下数据来自 shared-balance focused smoke，只表示该试验中观察到的变化；若要将它作为 `LOCAL_REPLICA` round-robin 的严格 A/B 证据，还需在运行日志中显式记录并确认 shared-expert mode。在均匀 `8×8192 input / 32 output` smoke 中：

| 指标 | 优化前 | Shared balance 后 | 变化 |
|---|---:|---:|---:|
| Total throughput | 15,670.8 tok/s | 17,867.1 tok/s | +14.0% |
| Mean ITL | 32.72 ms | 28.74 ms | -12.2% |

### 4.4 Decode-aware EPLB（历史 prototype，已从新 PR 删除）

旧实验分支曾实现以下能力；这些修改不在 #2110 中，当前 upstream EPLB 行为保持不变：

- `load_mode="prefill"`：保持原行为。
- `load_mode="decode"`：只使用 Decode 专家负载。
- `load_mode="all"`：同时统计 Prefill 和 Decode。
- `max_rebalances`：完成指定次数后冻结 placement。
- Decode CUDAGraph padding mask：dummy/padding token 不进入专家负载统计。
- 复用已有 DP metadata 同步结果，不为 EPLB 每步额外增加 collective。
- 首次重排必须等待完整 `load_window_size`，避免用最先结束的少量 trajectory 形成偏置 placement。
- 冻结后通过固定地址 GPU flag 跳过已捕获图中的 histogram/atomic 统计，同时保留 expert remap。

历史实验配置：

```bash
--eplb-enable \
--eplb-config '{
  "load_mode":"decode",
  "load_window_size":1000,
  "rebalance_interval":3000,
  "max_rebalances":1,
  "rebalance_layers_per_chunk":64,
  "rebalance_min_balancedness":0.9,
  "num_redundant_experts":0,
  "placement_policy":"naive"
}'
```

实验策略是“一次学习、一次迁移、然后冻结”，而不是持续在线搬运专家。严格 A/B 仍显示 EPLB-off 更适合当前短窗口，因此没有把这套 prototype 带入新 PR。

### 4.5 DP Session Affinity

AgentX 是多轮会话。不开 session affinity 时，同一会话的后续 turn 可能被送到不同 DP rank；前一轮写入某张卡的 prefix KV，下一轮无法复用，整个 workload 会退化成反复冷 Prefill。

启用：

```bash
export ATOM_DP_SESSION_AFFINITY=1
```

路由策略变为：

1. 新 session 根据当前 token/request 压力选择 owner rank。
2. 后续 turn 固定返回该 owner，保持 prefix-cache locality。
3. AIPerf 使用 correlation ID 传递 session 身份。

相关说明与实现：

- [`ATOM/docs/distributed_guide.md`](ATOM/docs/distributed_guide.md)
- [`ATOM/recipes/DeepSeek-V4-Agentic-InferenceX.md`](ATOM/recipes/DeepSeek-V4-Agentic-InferenceX.md)
- [`ATOM/atom/model_engine/engine_core_mgr.py`](ATOM/atom/model_engine/engine_core_mgr.py)

### 4.6 TBO 安全保护

当前 native RCCL/fallback 路径是同步实现。DPA rank 上两个 TBO ubatch 可能以不同顺序进入全组 collective，因此可能死锁。当前配置会对 RCCL backend 关闭 TBO；没有 MoRI 时也会自动保护 DPA + EP fallback。

这保证了稳定性，但也意味着当前实现还没有获得 MoRI 的异步通信与计算重叠收益。

## 5. 无效或负收益方案

这些实验对后续调优很重要，因为它们限定了真正值得投入的方向。

### 5.1 短上下文 c128 Decode EPLB

测试：8×MI355X、真实 V4-Pro、DPA8 + EP8、256 input / 256 output、c128、每轮 256 请求。

| 配置 | 平均 Output tok/s | 相对 baseline | 平均 TPOT |
|---|---:|---:|---:|
| EPLB off | 3,623 | — | 30.22 ms |
| EPLB，0 冗余专家 | 3,420 | -5.6% | 32.91 ms |
| EPLB，64 冗余，naive | 3,358 | -7.3% | 33.83 ms |

- 0 冗余首次迁移约 3.58 秒。
- naive + 64 首次迁移约 3.91 秒，每卡额外占用约 16 GiB 权重。
- `biased + 64` 在首次在线迁移时卡住，已中止。
- 结论：此场景不应通过增加冗余专家直接打榜。

原始数据位于 [`benchmark_results/eplb_decode_20260830`](benchmark_results/eplb_decode_20260830)。

### 5.2 长上下文 c96 在线 EPLB

120 秒诊断窗中：

| 配置 | 完成请求 | Total tok/s | Output tok/s | Mean ITL |
|---|---:|---:|---:|---:|
| EPLB off | 152 | 31,285.8 | 129.23 | 20.20 ms |
| Decode EPLB 在线重排 | 145 | 30,328.2 | 119.24 | 22.22 ms |
| Prefill warmup 学习后冻结 | 141 | 29,520.8 | 111.04 | 22.67 ms |

在线 Decode 重排的整场 Output throughput 下降 7.7%，但对迁移后的相同请求配对分析显示：

- TTFT 改善 8.15%。
- E2E latency 改善 5.50%。
- 平均 ITL 改善 1.34%。
- 加权 Decode time/token 改善 0.79%。

这说明 learned placement 有小幅收益，但约 2.85 秒的在线迁移停顿会吃掉短窗口收益。用 Prefill warmup 学习 placement 更差，说明 Prefill expert 分布不能代替 Agentic Decode expert 分布。

`rebalance_layers_per_chunk=1` 也不合适：61 层迁移被拉长到约 9 秒，某层出现约 1.4 秒 stall，吞吐比整块迁移更差。

2026-09-01 又在同一代码、AITER、AgentX trace 和 c96 配置下做了一组严格开关对照。两轮均使用 120 秒发送窗口和 120 秒 grace，除 EPLB 开关及其配置外保持一致；EPLB-on 在正式窗口约 58 秒时完成一次重排，迁移约 3.32 秒后冻结。两轮平均输入长度几乎相同（64,776 vs 64,791 tokens），且都没有请求错误：

| 配置 | 完成请求 | Total tok/s | Output tok/s | Output tok/s/chip | Mean ITL | P90 ITL | P90 Interactivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| EPLB off | 177 | 48,110.33 | 338.04 | 42.26 | 18.29 ms | 22.14 ms | 45.16 tok/s/user |
| Decode EPLB，重排一次后冻结 | 159 | 43,260.50 | 336.67 | 42.08 | 20.30 ms | 27.14 ms | 36.84 tok/s/user |
| EPLB off 相对变化 | +11.3% | +11.2% | +0.4% | +0.4% | -9.9% | -18.4% | +22.6% |

因此，在这段短窗口里关闭 EPLB 更好：输出吞吐没有损失，prefill/总吞吐和交互延迟反而改善。主要原因是 120 秒窗口包含 EPLB 的在线统计、约 3.3 秒迁移和 remap 成本，迁移后的 placement 尚没有足够长的稳态时间摊销这些开销。这个结果支持“默认关闭 EPLB，除非能离线加载已校准 placement”的方向；但不能用 120 秒结果替代 3600 秒稳态结论，也不能把这里的 42.26 tok/s/chip 直接与一小时 EPLB-on 的 247.38 tok/s/chip 横向比较。

### 5.3 第一版动态 routed All-to-All

第一版原生 RCCL 按每层动态计算 split：

```text
route plan
  -> count All-to-All
  -> GPU→CPU split-size 同步
  -> pack
  -> forward All-to-All
  -> local fused MoE
  -> reverse All-to-All
  -> index_add combine
```

在 c96 形状的 8 卡通信微基准中：

```text
动态 routed RCCL：约 0.777 ms/MoE layer
旧 AG/RS fallback：约 0.090 ms/MoE layer
```

动态实现慢约 8.7 倍。端到端诊断中 Output throughput 从 fallback 的 129.23 tok/s 降至 26.72 tok/s，Mean ITL 从 20.20 ms 增至 87.71 ms。该版本只能作为 correctness prototype，不能打榜。

### 5.4 多 collective 和 custom collective 实验

- 将 hidden、IDs、weights 拆成三个 AllGather：小 smoke 看似更快，但 AgentX c96 出现严重长尾，已撤回。
- 使用 ATOM custom collective 传 packed payload：小 smoke 约提升 6%，但 AgentX c96 明显回退，已撤回。
- Python raw RCCL send/recv 和 symmetric-memory 原型没有显示出可持续收益。

最终保留的是单 packed collective 的 graph-safe Decode 路径。

## 6. Session Affinity 的受控 A/B

这是本轮最清晰、最可复现的单变量收益。两轮均为 c48、DPA8 + EP8 + RCCL hybrid + EPLB、完整 warmup + 3600 秒 profile，唯一变化是 `ATOM_DP_SESSION_AFFINITY=1`。

| 指标 | 无 Affinity | 开启 Affinity | 变化 |
|---|---:|---:|---:|
| 完成请求 | 2,314 | 3,843 | +66.1% |
| Request throughput | 0.6357 req/s | 1.0558 req/s | +66.1% |
| Input throughput | 79,559 tok/s | 147,131 tok/s | +84.9% |
| Output throughput | 542.16 tok/s | 1,040.90 tok/s | +92.0% |
| Total throughput | 80,101 tok/s | 148,171 tok/s | +85.0% |
| Mean TTFT | 20.38 s | 10.29 s | -49.5% |
| P50 TTFT | 14.63 s | 8.81 s | -39.8% |
| Mean ITL | 49.14 ms | 20.12 ms | -59.1% |
| P90 ITL | 89.35 ms | 24.42 ms | -72.7% |
| P90 Interactivity | 11.19 tok/s/user | 40.95 tok/s/user | +265.9% |
| AIPerf cache read | 73.76% | 96.53% | +22.77 pp |
| Server prefix hit，平均 | 57.64% | 91.61% | +33.97 pp |

Affinity 运行计数：

```text
dp_affinity_new=439
dp_affinity_owner_hit=3942
dp_affinity_spill=0
```

因此最终性能恢复的主因不是单纯的 MoE kernel，而是避免多轮 Agentic 会话在 DP rank 之间漂移，从而恢复 prefix KV cache locality。

## 7. 完整 AgentX c48–c128 结果

以下均开启 Session Affinity，并使用相同服务配置；每档独立重启。

| Concurrency | 成功请求 | Req/s | Input tok/s | Output tok/s | Total tok/s | Output tok/s/chip | P90 Interactivity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| c48 | 3,843 | 1.0558 | 147,130.58 | 1,040.90 | 148,171.48 | 130.11 | 40.95 |
| c64 | 5,320 | 1.4615 | 175,554.40 | 1,441.53 | 176,995.93 | 180.19 | 30.09 |
| c96 | 7,889 | 2.1673 | 223,386.25 | 1,979.01 | 225,365.26 | 247.38 | 19.07 |
| c128 | 8,723 | 2.3964 | 247,478.41 | 2,239.51 | 249,717.92 | 279.94 | 16.41 |

延迟与缓存：

| Concurrency | Mean TTFT | P50 TTFT | Mean ITL | P90 ITL | AIPerf cache read | Server prefix hit，平均 |
|---:|---:|---:|---:|---:|---:|---:|
| c48 | 10.29 s | 8.81 s | 20.12 ms | 24.42 ms | 96.53% | 91.61% |
| c64 | 9.94 s | 8.38 s | 25.39 ms | 33.23 ms | 95.64% | 91.01% |
| c96 | 9.52 s | 7.69 s | 38.80 ms | 52.43 ms | 94.78% | 90.42% |
| c128 | 10.14 s | 7.84 s | 46.42 ms | 60.96 ms | 94.77% | 89.69% |

相邻并发增量：

| 区间 | Total throughput 变化 | P90 Interactivity 变化 |
|---|---:|---:|
| c48 → c64 | +19.45% | -26.51% |
| c64 → c96 | +27.33% | -36.62% |
| c96 → c128 | +10.81% | -13.98% |

推荐选择：

- P90 Interactivity 要求不低于 30 tok/s/user：选择 c64。
- 兼顾较高吞吐和可接受交互性：选择 c96。
- 只追求最大吞吐：选择 c128，但已出现明显边际收益下降。

## 8. 与官方 ATOM DPA CI 对比

官方来源：[InferenceX run #33376779864](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33376779864)。

两边均为 MI355X、DeepSeek-V4-Pro、AgentX、MTP、DPA，并按相同 concurrency 对齐。主要拓扑差异是：

```text
官方 CI：DPA + EP1
历史本地 DEP：DPA8 + EP8 + RCCL hybrid + EPLB + Session Affinity
```

因此这是高质量的系统级对照，但不是只改变一个开关的纯代码 A/B。

| c | CI Output/chip | DEP Output/chip | Output 提升 | CI P90 Interactivity | DEP P90 Interactivity | 交互性提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 124.75 | 130.11 | +4.30% | 37.52 | 40.95 | +9.13% |
| 64 | 170.40 | 180.19 | +5.75% | 26.38 | 30.09 | +14.05% |
| 96 | 228.62 | 247.38 | +8.21% | 18.70 | 19.07 | +1.99% |
| 128 | 266.80 | 279.94 | +4.92% | 16.00 | 16.41 | +2.55% |

Total throughput 和 P50 TTFT 也全部改善：

| c | Total throughput 提升 | P50 TTFT 改善 |
|---:|---:|---:|
| 48 | +4.08% | -10.95% |
| 64 | +4.81% | -10.79% |
| 96 | +7.38% | -13.67% |
| 128 | +4.11% | -13.75% |

四个本地 DEP 点都位于对应官方 CI 点的右上方。c96 的吞吐收益最大，c64 的 P90 Interactivity 收益最大。

## 9. 与 B200 vLLM / SGLang CI 对比

来源：

- [vLLM B200 CI job #96964146599](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32526153588/job/96964146599)
- [SGLang B200 CI job #98793577271](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33154296172/job/98793577271)

这两个 B200 job 都是 DPA8 + EP8，但都**没有开启 EPLB**：

| 配置项 | vLLM B200 job | SGLang B200 job | 本地 ATOM c96 最终跑分 |
|---|---|---|---|
| DPA / EP | DP8 + EP8 | DP8 + EP8 | DPA8 + EP8 |
| EPLB | `enable_eplb=false` | `enable_eplb=False` | `eplb_enable=True` |
| EPLB replicas | `num_redundant_experts=0` | `ep_num_redundant_experts=0` | `num_redundant_experts=0` |
| EPLB 运行方式 | 未运行 | 未运行 | Decode 统计，重排一次后冻结 |
| MoE transport/backend | `allgather_reducescatter` + `deep_gemm_amxf4_mega_moe` | `megamoe` | native RCCL hybrid + standard MoE |
| Shared expert | job 未显式给出 fusion 开关 | 显式 `--disable-shared-experts-fusion` | 实测为 standalone，未进 EPLB |
| DP 请求分配 | 外部 router `consistent_hash` | `total_requests` | `least_tokens` + session affinity |
| KV offload | `vllm-simple` | HiCache | 无 |

因此，当前只在模型、AgentX workload、MTP-3 和 DPA8 + EP8 大拓扑上对齐；EPLB、MoE backend、shared-expert 路径、DP 调度和 KV offload 都未对齐。因此后面的成绩只能作为跨平台目标线，不是 EPLB 或单一实现差异的 A/B。

高并发共同点如下，每个单元格为 `Output tok/s/chip / P90 Interactivity`：

| c | DEP · MI355X | vLLM · B200 | SGLang · B200 |
|---:|---:|---:|---:|
| 64 | 180.19 / 30.09 | 224.07 / 42.99 | 220.42 / 37.74 |
| 96 | 247.38 / 19.07 | 338.78 / 36.32 | 294.07 / 30.67 |
| 128 | 279.94 / 16.41 | 403.29 / 32.73 | 323.22 / 28.02 |

B200 vLLM 的高并发扩展能力最强；SGLang 在 c128 达到 323.22 tok/s/chip，但 c160 回落到 300.24 tok/s/chip、16.78 tok/s/user，已进入过载区。vLLM 在 c160 仍达到 437.54 tok/s/chip、28.43 tok/s/user，c196 的峰值吞吐为 449.91 tok/s/chip，但交互性降至 19.90 tok/s/user。

这部分只能作为跨平台目标线，不能直接归因到框架：

- DEP 使用 AMD MI355X。
- vLLM 使用 DGX B200 和 `vllm-simple` KV offload。
- SGLang 使用 nscale B200 和 HiCache。
- 镜像、硬件、KV-offload backend 和调度参数都不同。

## 10. 稳定性与验证

最终 c48–c128 四档均完成 3600 秒 profile：

- c48、c64：warmup 和 profile 均无请求错误。
- c96：2 个客户端写请求错误，约 0.025%。
- c128：warmup 1 个、profile 2 个客户端连接错误。
- 错误均为客户端 `ClientOSError` / connection reset。
- 服务端没有 traceback、HIP illegal memory、RCCL error 或 collective hang。
- 所有档位的 affinity spill 均为 0。
- 所有档位的 EPLB 均只重排一次并正常冻结。

实现验证包括：

- Ruff、`compileall`、`git diff --check`：通过。
- Host 定向测试：33 passed，4 skipped。
- ROCm/AITER 容器测试：78 passed。
- RCCL/shared-expert focused suite：28 passed。
- 2-rank dispatch → fused MoE → reverse RCCL → combine：通过。
- 8-rank top-k routed roundtrip：通过。
- V4 Decode graph capture：9 个 bucket 通过。
- 真实 8×8192/32、8×8192/128 和完整 AgentX：通过。

当前 RCCL-only 代码已提交到 #2110；该 PR 的 ROCm/AITER focused suite 为 134 passed，且 Black、Ruff、compileall、`git diff --check` 均通过。

## 11. 后续优化优先级

### P0：保持 Session Affinity

所有 Agentic DPA 测试都应设置：

```bash
export ATOM_DP_SESSION_AFFINITY=1
```

并确保客户端发送 `X-Dynamo-Session-ID` 或可回退的 `X-Correlation-ID`。这是当前最大的已验证收益，不能省略。

### P1：把 EPLB 变成离线静态 placement

推荐流程：

1. 使用与榜单相同 concurrency、相似输出长度的 Decode 流量校准。
2. 等待一次 EPLB rebalance 完成并冻结。
3. 保存 logical-to-physical placement。
4. 下次启动时在 CUDAGraph capture 前直接加载 placement。
5. 对无冗余专家的场景，将 remap 烘焙到 `expert_map`，移除运行时 remap kernel。

当前 remap 约为 12.7–15.6 μs/层，61 层累计约 0.8–0.95 ms/forward；静态化后可以直接消除这部分固定成本。

### P2：实现真正 GPU-resident routed transport

要继续接近 MoRI，需要将以下步骤全部留在 GPU：

- destination count 与 prefix sum。
- token/hidden/ID/weight 的 fused pack。
- 固定或对称 staging buffer。
- FP8/FP4 wire format。
- fused inverse permutation、routing weight 与 Top-K reduce。
- graph-safe dispatch/combine。

简单调用动态 `all_to_all_single` 不够，因为 CPU split-size 同步和多个小 kernel 会在 61 个 MoE 层上反复累积。

### P3：恢复安全的 TBO overlap

当前 native RCCL backend 为保证 collective 顺序而关闭 TBO。下一步需要给两个 ubatch 独立的 communicator、buffer 或严格的全 rank 调度序列，再验证：

```text
ubatch 0 communication <-> ubatch 1 expert compute
```

这是追平 MoRI 的关键能力之一。

### P4：继续优化 Prefill 和高并发调度

完整结果显示 c128 相对 c96 的 Total throughput 只增加 10.81%，同时 P90 Interactivity 下降 13.98%，已经接近当前系统的饱和区。后续应重点优化：

- 冷 session 的 Prefill/TTFT。
- 长短请求在 DPA rank 之间的负载平衡。
- Prefix-cache-aware admission。
- Prefill coalescing 和 phase alignment。
- XGMI topology-aware dispatch。

单纯把 concurrency 从 c128 继续提高，不太可能获得更好的综合 Pareto 点。

## 12. 最终结论

这轮工作的核心成果不是“把一个 All-to-All API 换成另一个 API”，而是形成了一条能实际运行和打榜的非 MoRI DEP 路径：

```text
DPA8 分散 Attention/KV
  + EP8 分片专家
  + RCCL hybrid 通信
  + EPLB-off shared-expert 均衡
  + session affinity 保持多轮 KV locality
```

最终结果证明：

- 相比同配置关闭 Session Affinity，c48 总吞吐提升约 85%，P90 Interactivity 提升约 266%。
- 相比最新官方 ATOM DPA CI，c48–c128 的 Output/chip、P90 Interactivity、Total throughput 和 P50 TTFT 全部改善。
- 当前实测下 EPLB-off 更快；Decode EPLB prototype 已从新 PR 删除，历史数据仅保留为后续离线 placement 研究依据。
- 当前最实用的榜单点是 c64 或 c96；c128 适合只追求吞吐的场景。
- 与 B200 vLLM 的剩余差距主要属于通信系统和调度系统差距，后续应投入 GPU-resident routed transport、低精度 wire、融合 pack/combine 与 TBO overlap。

## 13. 原始结果与参考链接

本地完整结果：

- [c48，无 Session Affinity](../001-dsv4-perf/runs/atom-native-rccl-hybrid-eplb-dpa8-ep8-agentx-c48-ci3600-20260831/aiperf_artifacts/profile_export_aiperf.json)
- [c48，开启 Session Affinity](../001-dsv4-perf/runs/atom-native-rccl-hybrid-eplb-affinity-dpa8-ep8-agentx-c48-ci3600-20260831/aiperf_artifacts/profile_export_aiperf.json)
- [c64，开启 Session Affinity](../001-dsv4-perf/runs/atom-native-rccl-hybrid-eplb-affinity-dpa8-ep8-agentx-c64-ci3600-20260831/aiperf_artifacts/profile_export_aiperf.json)
- [c96，开启 Session Affinity](../001-dsv4-perf/runs/atom-native-rccl-hybrid-eplb-affinity-dpa8-ep8-agentx-c96-ci3600-20260831/aiperf_artifacts/profile_export_aiperf.json)
- [c128，开启 Session Affinity](../001-dsv4-perf/runs/atom-native-rccl-hybrid-eplb-affinity-dpa8-ep8-agentx-c128-ci3600-20260831/aiperf_artifacts/profile_export_aiperf.json)
- [短上下文 EPLB A/B](benchmark_results/eplb_decode_20260830)

外部 CI：

- [官方 ATOM MI355X DPA CI](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33376779864)
- [vLLM B200 CI，attempt 3](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32526153588/attempts/3)
- [SGLang B200 CI](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33154296172/job/98793577135)
