# Assignment 02 · Tensor Core & Pipeline

Session 3（Tensor Core：从 mma.sync 到 tcgen05）与 Session 4（tiling、TMA、
pipeline）的配套作业。具体题目见 `handout/assignment02.pdf`；PDF 由
`handout/src/assignment02.md` 生成，以 Markdown 源文件为准。

## 目录结构

```
handout/assignment02.pdf  作业本体 START HERE
cuda/                 模块 0-5 的 CUDA 练习
  common.h            错误检查、计时、对拍的公共工具
  Makefile            编译(注意 ARCH 的用法,见下)
  m5_lowprec/         模块 5:低精度与 block scaling
kernels/              模块 5 的 Python 练习(5.1 / 5.2)
tests/                pytest 判测
team/                 团队选做题 C1 / C2 的材料与任务书
```

## 硬件

- 模块 0-1 在 RTX 5090 或 B300 上可完成
- 模块 2 全部无卡可判
- 模块 3-4 以及 5.3-5.4 需要 B300（sm_100 家族）；5.1-5.2 是 host
  Python 实验，不需要 GPU

- 各题的卡要求在题面标注

## 构建

CUDA 部分统一用显式 `-gencode`,Makefile 已配好,默认 `ARCH=100f`
(覆盖 B200/B300 全家族):

```bash
cd assignment02/cuda
make run/m5_lowprec/03a_encode_check      # 编译并运行单个练习
ARCH=89 make bin/...                      # 老架构按需指定
```

注意:`nvcc -arch=sm_103a` 这类简写在 CUDA 13.0 下会把用户代码的 PTX
target 展开成不带后缀的 compute_103,arch-specific 指令(例如 e2m1 的
cvt)会被 ptxas 拒绝,报错信息还会误导为"指令不支持"。显式写
`-gencode arch=compute_100f,code=sm_100f` 没有这个问题。这就是 Makefile
不用 `-arch` 简写的原因。

## Python 部分

```bash
cd assignment02
uv sync && uv run pytest tests/
```

M6 另外使用固定的 TileLang 版本：

```bash
uv sync --extra tilelang
```

## 关于 AI

必做题沿用系列惯例(仓库根目录 CLAUDE.md):AI 可以帮你理解、review、
解读报错,但每道题的实测数据必须来自你自己跑的卡,解释必须是你自己
能答辩的。团队题(C1/C2)不设限制,怎么用 AI 都可以;答辩时问的是
你们的决策和证据,答不上来的部分不算数。
