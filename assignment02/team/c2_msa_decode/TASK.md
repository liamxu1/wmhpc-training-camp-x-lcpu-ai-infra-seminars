# C2:MiniMax M3 MSA decode——小 batch 这一半还空着

## 背景(2026-08 上游现状,vllm 主树 pin `d4da0c5`)

MiniMax M3 的稀疏注意力(MSA):lightning indexer 选 top-k=16 个 KV 块,
块大小 = KV page = 128,主注意力只在选中的块上做(GQA,head_dim=128,
最多 64 个 query head / 4 个 KV head)。vLLM 里的现状(本目录
`vllm_msa_ref/` 快照):

- prefill 走 CUTLASS `fmha_sm100`;
- decode 默认走 Triton:split-K 部分和 + LSE merge 两个 kernel
  (`sparse_attn.py`);
- 上游近期补了一个 opt-in 的 CUTLASS decode 路径
  (`msa_cutlass_sparse_decode.py`),但官方自己的 kernel benchmark 把
  启用门槛定在 batch>=16(代码原话:"Kernel benchmarks put the CUTLASS
  crossover at 16 requests for TP1 and TP4"),且形状受限。

也就是说:延迟敏感的小 batch decode(线上最常见的 regime)至今仍由
Triton kernel 承担,而"batch<16 时 CUTLASS 反而不划算"这个 crossover
本身,就是这道题要研究的现象——它与 assignment 4.5(瘦 GEMM 达成率
塌方)是同一件事在 attention 上的版本。

## 任务(三层)

1. 测量:先 profile Triton 基线(小 batch decode，batch $\in \{1,4,8,16\}$),
   瓶颈是哪一项?测完再设计——这个顺序是任务要求,不许倒过来。
2. 分析:讨论点逐个"结论 + 证据"。
3. 挑战,二选一:
   (a) 动手做小 batch regime 的 CUDA decode(或改良 Triton),正确性
       验收方案自定(见讨论点 6);性能对 Triton 基线,并对照上游
       CUTLASS 路径在 batch>=16 的表现,回答 crossover 为什么在 16;
   (b) 论证"不值得做":收益上限 < 工程成本的完整证据链(收益上限
       从 roofline 推,工程成本从 CUTLASS 路径的复杂度估)。

交付:代码 + 报告 + 答辩。验收方案先贴出来接受别组挑战,再动工。

## 讨论点

1. top-k=16 块 × 块大小 128 的形状下,算 decode 一步的 arithmetic
   intensity:tensor core 有没有用武之地?(联动 4.2 与 4.5 的结论。)
2. 两个 kernel(部分和 + merge)融不融?merge 放 cluster / mbarrier
   里做的可行性?
3. top-k 块要经 block table 两级间接寻址,TMA tensor map 能不能表达
   这种访问?不能的话用什么搬?(文档考证 + 实验验证。)
4. FP8 KV cache 的 scale 处理放哪一层?(`test_sparse_attn_fp8_scale.py`
   是上游的口径。)
5. 先 profile Triton 基线,瓶颈是哪一项——用数据说话,测完再设计。
6. 验收方案自己定:对什么参照、tolerance 怎么定?方案先写出来给别的
   组挑毛病。

## 材料

- `vllm_msa_ref/`:vllm 主树 `vllm/models/minimax_m3/` 关键文件快照,
  pin commit `d4da0c5`:
  - `sparse_attn.py`:Triton decode 基线(split-K + LSE merge)与其
    调用约定,块布局注释在文件头
  - `index_topk.py`:lightning indexer 的 top-k 选择(两级间接寻址的
    上游实现)
  - `msa_cutlass_sparse_decode.py`:新的 CUTLASS decode 路径与
    crossover 门槛(`_MIN_CUTLASS_BATCH_SIZE = 16`)
  - `sparse_attention_msa.py`:backend dispatch(哪个 regime 走哪条路)
  - `test_sparse_attn_fp8_scale.py`:上游测试,FP8 scale 口径
- `harness/`:独立运行脚手架(合成 top-k 索引 + paged KV 生成器、
  SDPA 稠密参考、Triton 基线独立化入口)。
- 形状表:topk=16,page=128,head_dim=128,q heads 64 / kv heads 4
  (TP 会切分 head 数;上游常量见 `msa_cutlass_sparse_decode.py` 头部)。
