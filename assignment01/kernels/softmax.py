"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_ym,
    BLOCK_N: tl.constexpr,
):
    # ---- program id = row index ----
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m

    # ---- column offsets ----
    offs_n = tl.arange(0, BLOCK_N)

    # ---- load ----
    x_ptrs = x_ptr + offs_m * stride_xm + offs_n
    x_mask = offs_n < N
    x = tl.load(x_ptrs, mask=x_mask, other=-float("inf"))

    # ---- row max ----
    row_max = tl.max(x, axis=0)

    # ---- subtract max + exp ----
    x = x - row_max
    e = tl.exp(x)

    # ---- row sum ----
    row_sum = tl.sum(e, axis=0)

    # ---- normalize ----
    y = e / row_sum

    # ---- store ----
    y_ptrs = y_ptr + offs_m * stride_ym + offs_n
    tl.store(y_ptrs, y, mask=x_mask)
    

def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    M, N = x.shape
    y = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)
    softmax_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        y.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return y
