#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#define M 16
#define N 8
#define K 32

#include "common.h"

// ============================== CPU REFERENCE ==============================
void cpu_gemm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* D) {
    // A: 16x32 row-major, B: 32x8 row-major
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[k * N + n]);
            }
            D[m * N + n] = sum;
        }
    }
}

// ============================== DEVICE KERNEL ==============================
__global__ void fp8_mma_kernel(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* D) {
    int lane = threadIdx.x;
    int gid = lane >> 2;   // quad ID, 0..7
    int tig = lane & 3;    // thread-in-quad, 0..3

    // ---- A fragment: 4 个 b32 ----
    unsigned ra[4];
    // 每个 lane 的 16 个 fp8 分布在 ra[0..3] 中
    // 根据 PTX: ra[0], ra[1] 对应 row=gid 的 K 前半/后半
    //           ra[2], ra[3] 对应 row=gid+8 的 K 前半/后半
    // 每个 b32 里 4 个 fp8 按 tig 交错

    __nv_fp8_e4m3* p = reinterpret_cast<__nv_fp8_e4m3*>(ra);
    p[0] = A[(gid) * 32 + tig * 4];
    p[1] = A[(gid) * 32 + tig * 4 + 1];
    p[2] = A[(gid) * 32 + tig * 4 + 2];
    p[3] = A[(gid) * 32 + tig * 4 + 3];
    
    p[4] = A[(gid + 8) * 32 + tig * 4];
    p[5] = A[(gid + 8) * 32 + tig * 4 + 1];
    p[6] = A[(gid + 8) * 32 + tig * 4 + 2];
    p[7] = A[(gid + 8) * 32 + tig * 4 + 3];
    
    p[8] = A[(gid) * 32 + tig * 4 + 16];
    p[9] = A[(gid) * 32 + tig * 4 + 17];
    p[10] = A[(gid) * 32 + tig * 4 + 18];
    p[11] = A[(gid) * 32 + tig * 4 + 19];
    
    p[12] = A[(gid + 8) * 32 + tig * 4 + 16];
    p[13] = A[(gid + 8) * 32 + tig * 4 + 17];
    p[14] = A[(gid + 8) * 32 + tig * 4 + 18];
    p[15] = A[(gid + 8) * 32 + tig * 4 + 19];

    // ---- B fragment: 2 个 b32 ----
    unsigned rb[2];

    // 每个 lane 的 8 个 fp8 分布在 rb[0..1] 中
    // rb[0] 对应 col=gid 的 K 前半
    // rb[1] 对应 col=gid 的 K 后半

    p = reinterpret_cast<__nv_fp8_e4m3*>(rb);
    p[0] = B[(tig * 4) * 8 + gid];
    p[1] = B[(tig * 4 + 1) * 8 + gid];
    p[2] = B[(tig * 4 + 2) * 8 + gid];
    p[3] = B[(tig * 4 + 3) * 8 + gid];

    p[4] = B[(tig * 4 + 16) * 8 + gid];
    p[5] = B[(tig * 4 + 17) * 8 + gid];
    p[6] = B[(tig * 4 + 18) * 8 + gid];
    p[7] = B[(tig * 4 + 19) * 8 + gid];

    // ---- 累加器 ----
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];

    // ---- mma 指令（修正：4 个 A 寄存器）----
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );

    // ---- D 写回 ----
    D[gid * 8 + tig * 2] = d[0];
    D[gid * 8 + tig * 2 + 1] = d[1];
    D[(gid + 8) * 8 + tig * 2] = d[2];
    D[(gid + 8) * 8 + tig * 2 + 1] = d[3];
}

// ============================== MAIN ==============================
int main(int argc, char** argv) {
    int seed = 0;
    if (argc > 1) seed = atoi(argv[1]);

    // 用可精确表示为 e4m3 的小整数生成数据
    // e4m3 精确表示范围: ±0, ±1, ±2, ±3, ±4, ±6, ±8, ±10, ±12, ±14, ±16, ...
    // 我们用 0..7 的整数，保证精确
    srand(seed);

    __nv_fp8_e4m3 hA[M * K];
    __nv_fp8_e4m3 hB[K * N];
    float ref[M * N];

    for (int i = 0; i < M * K; ++i) {
        int val = rand() % 8;  // 0..7, e4m3 精确
        hA[i] = __nv_fp8_e4m3((float)val);
    }
    for (int i = 0; i < K * N; ++i) {
        int val = rand() % 8;
        hB[i] = __nv_fp8_e4m3((float)val);
    }

    // CPU 参考
    cpu_gemm_fp8(hA, hB, ref);

    // GPU
    __nv_fp8_e4m3 *dA, *dB;
    float *dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(ref)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0, sizeof(ref)));

    fp8_mma_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();

    float got[M * N];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    // 严格比较
    int bad = 0;
    for (int i = 0; i < M * N; ++i) {
        if (got[i] != ref[i]) {
            if (bad < 4)
                printf("MISMATCH D[%d]: got %.0f, want %.0f\n", i, got[i], ref[i]);
            bad++;
        }
    }

    if (bad == 0)
        printf("PASS\n");
    else
        printf("FAIL: %d / %d mismatches\n", bad, M * N);

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad != 0;
}