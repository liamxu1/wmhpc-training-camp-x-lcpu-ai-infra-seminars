---
title: 语法样例(非发布物)
subtitle: handout 管线自检
---

# 示例模块

模块引言一段。行内代码 `make run/m0_env/01_first_mma`,粗体**必须实测**,
数学 $AI = 2MNK / (2MK + 2NK + 2MN)$,下划线路径 `cuda/m4_gemm/01_tiled.cu`。

::: reading
PTX ISA 9.7.14；课件 S026--S030；`assignment02/README.md`。
:::

### 0.1 {.prob type=HANDS-ON file=cuda/m0_env/01_first_mma.cu}

编译运行最小 mma 程序:

```
cd assignment02/cuda
make run/m0_env/01_first_mma
```

### 0.2 {.prob type=DERIVE}

推导峰值,填表:

| 量 | 5090 | B300 |
|---|---|---|
| bf16 FLOP/cyc/SM | | |
| 峰值 TFLOPS | | |

### 0.3 {.prob type=CONCEPT opt=Optional}

判断对错:

(a) 第一条。
(b) 第二条。

::: {.capstone title="prob 9.9(FROM-SCRATCH):示例压轴" file=cuda/demo.cu}
压轴题题面,框内也可以有代码块:

```
make run/demo
```
:::

::: lookback
模块回望的评注文字。
:::
