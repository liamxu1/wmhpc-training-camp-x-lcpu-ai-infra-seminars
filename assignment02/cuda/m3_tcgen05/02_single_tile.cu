// 问题 3.2(模块压轴):从零写 tcgen05 单 tile GEMM。
//
// 形状 m128n64k64,bf16 输入,f32 累加,cta_group::1,单 block 128 线程。
// 数据通路:global -> smem(K-major + 128B swizzle)-> tcgen05.mma ->
// TMEM -> tcgen05.ld -> global。判测(main 已给出)用小整数严格对拍。
//
// 给你的材料:课件 F27 的七步流程(下面 kernel 里只留了步骤注释)、
// 你在 2.2 写的 descriptor 编码(SM100 位域)、2.3 的 swizzle_128B
// (staging 布局用它;布局错,结果必错——这里是它的真硬件判测)。
// 其余(TMEM alloc、mbarrier、idesc、tcgen05.mma/ld 的写法)自己查
// PTX ISA 对应章节,课件 C15-C21 讲过每一件的语义,数字换成本题形状。
//
// 两个提醒,直接说明:
// - smem 写完到发射 mma 之间需要 fence.proxy.async(2.1 排序题的答案
//   在这里上真硬件;漏掉的现象自己观察一次,写进报告)
// - tcgen05.ld 每个 warp 只能读自己的 32 条 lane(3.1(a));taddr 高
//   16 bit 是 lane 偏移、低 16 bit 是列偏移;ld 之后要 tcgen05.wait::ld
//
// 运行:make run/m3_tcgen05/02_single_tile;多 seed:./judge_tile.sh
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__global__ void tcgen05_tile(const __nv_bfloat16* gA,
                             const __nv_bfloat16* gB,
                             float* gD) {
    constexpr int TMEM_COLS = 64;

    constexpr int A_BYTES = M * K * sizeof(__nv_bfloat16); // 128*64*2 = 16384
    constexpr int B_BYTES = N * K * sizeof(__nv_bfloat16); //  64*64*2 =  8192

    constexpr int A_OFF = 0;
    constexpr int B_OFF = A_BYTES;

    // A + B staging.
    // A: 16 KB
    // B:  8 KB
    __shared__ __align__(128) unsigned char smem[A_BYTES + B_BYTES];

    // MMA completion barrier.
    __shared__ __align__(8) uint64_t mma_mbar;

    // tcgen05.alloc writes the TMEM base address here.
    __shared__ __align__(4) uint32_t tmem_addr;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const uint32_t smem_base =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(smem));

    const uint32_t a_smem = smem_base + A_OFF;
    const uint32_t b_smem = smem_base + B_OFF;

    const uint32_t mbar_addr =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(&mma_mbar));

    const uint32_t tmem_addr_smem =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(&tmem_addr));

    // ============================================================
    // (1) mbarrier initialization + TMEM allocation
    // ============================================================

    // Only thread 0 initializes the mbarrier.
    if (tid == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            :
            : "r"(mbar_addr)
            : "memory");

        // Make the mbarrier initialization visible to async proxy.
        asm volatile(
            "fence.mbarrier_init.release.cluster;"
            ::: "memory");
    }

    // tcgen05.alloc is sync.aligned, so the whole warp executes it.
    //
    // We use warp 0 as the allocation warp.
    if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned."
            "shared::cta.b32 [%0], %1;"
            :
            : "r"(tmem_addr_smem),
              "r"(TMEM_COLS)
            : "memory");
    }

    // Wait until:
    //   1. mbarrier is initialized
    //   2. tcgen05.alloc has written tmem_addr
    //   3. all threads see the staging setup
    __syncthreads();

    // The allocated TMEM base address.
    const uint32_t tmem =
        *reinterpret_cast<volatile uint32_t*>(&tmem_addr);

    // ============================================================
    // (2) Global -> SMEM
    //
    // Physical SMEM layout:
    //
    //   swz128(row, colByte)
    //
    // A is [M,K] = [128,64]
    // B is [N,K] = [ 64,64]
    //
    // One row contains 64 bf16 = 128 bytes.
    // ============================================================

    uint16_t* smem16 =
        reinterpret_cast<uint16_t*>(smem);

    const uint16_t* gA16 =
        reinterpret_cast<const uint16_t*>(gA);

    const uint16_t* gB16 =
        reinterpret_cast<const uint16_t*>(gB);

    // -------------------------
    // A: 128 x 64 bf16
    // -------------------------
    for (int idx = tid; idx < M * K; idx += blockDim.x) {
        const int row = idx / K;
        const int k   = idx % K;

        const int colByte = k * 2;
        const int off = swz128(row, colByte);

        smem16[(A_OFF + off) >> 1] = gA16[idx];
    }

    // -------------------------
    // B: 64 x 64 bf16
    // -------------------------
    for (int idx = tid; idx < N * K; idx += blockDim.x) {
        const int row = idx / K;
        const int k   = idx % K;

        const int colByte = k * 2;
        const int off = swz128(row, colByte);

        smem16[(B_OFF + off) >> 1] = gB16[idx];
    }

    // ============================================================
    // (3) Generic SMEM stores -> async proxy
    //
    // Required by the problem statement.
    // ============================================================

    asm volatile(
        "fence.proxy.async.shared::cta;"
        ::: "memory");

    // Make all normal SMEM stores visible to all 128 threads.
    __syncthreads();

    // Synchronize the tcgen05 async proxy with the thread sync.
    //
    // This is the ordering point used by SM100 tcgen05 kernels
    // before issuing MMA.
    asm volatile(
        "tcgen05.fence::after_thread_sync;"
        ::: "memory");

    // ============================================================
    // Instruction descriptor
    //
    // BF16 x BF16 -> FP32
    //
    // MMA_M = 128
    // MMA_N = 64
    // MMA_K = 16
    //
    // PTX fields:
    //   bit 4      : accumulator type FP32
    //   bit 7      : A type BF16
    //   bit 10     : B type BF16
    //   bits 17..23: MMA_N / 8
    //   bits 24..31: MMA_M / 16
    // ============================================================

    constexpr uint32_t IDESC =
        (1u << 4) |
        (1u << 7) |
        (1u << 10) |
        ((uint32_t(N) >> 3) << 17) |
        ((uint32_t(M) >> 4) << 24);

    // For this problem:
    //
    //   IDESC =
    //       1<<4
    //     | 1<<7
    //     | 1<<10
    //     | 8<<17
    //     | 8<<24
    //
    // because N=64 and M=128.
    //
    // This is exactly the standard BF16->FP32 tcgen05.f16
    // descriptor construction. 
    //
    // ============================================================

    // ============================================================
    // SMEM descriptor
    //
    // From problem 2.2:
    //
    // K-major + 128B swizzle
    //
    //   LBO    = 0
    //   SBO    = 1024
    //   layout = 2
    //
    // IMPORTANT:
    // Do NOT change these to the no-swizzle descriptor values.
    // These are the values validated by problem 2.2.
    // ============================================================

    auto make_desc = [](uint32_t addr) -> uint64_t {
        uint64_t d = 0;

        // start_address [0,14)
        d |= (uint64_t)((addr >> 4) & 0x3FFF);

        // LBO [16,30)
        // 128B-swizzle: problem 2.2 says LBO = 0.
        d |= 0ull << 16;

        // SBO [32,46)
        // 8 rows * 128 bytes = 1024 bytes.
        d |= (uint64_t)((1024u >> 4) & 0x3FFF) << 32;

        // version [46,48) = 1
        d |= 1ull << 46;

        // layout_type [61,64) = 128B swizzle = 2
        d |= 2ull << 61;

        return d;
    };

    // ============================================================
    // (4) Four K=16 MMA instructions
    //
    // K = 64
    // MMA_K = 16
    //
    // Therefore:
    //
    //   MMA #0: K [ 0,16)
    //   MMA #1: K [16,32)
    //   MMA #2: K [32,48)
    //   MMA #3: K [48,64)
    //
    // With 128B swizzle, the descriptor starts for successive
    // 16-K slices are:
    //
    //   + 0
    //   +32
    //   +64
    //   +96
    //
    // This is the physical 32-byte stride of one MMA-K tile
    // under the 128B-swizzled staging layout.
    //
    // Only one thread issues MMA.
    // ============================================================

    if (tid == 0) {

        const uint64_t a0 = make_desc(a_smem + 0);
        const uint64_t b0 = make_desc(b_smem + 0);

        const uint64_t a1 = make_desc(a_smem + 32);
        const uint64_t b1 = make_desc(b_smem + 32);

        const uint64_t a2 = make_desc(a_smem + 64);
        const uint64_t b2 = make_desc(b_smem + 64);

        const uint64_t a3 = make_desc(a_smem + 96);
        const uint64_t b3 = make_desc(b_smem + 96);

        // --------------------------------------------------------
        // MMA #0
        //
        // D = A0 * B0
        //
        // predicate = false:
        // do NOT use the old TMEM accumulator.
        // --------------------------------------------------------

        asm volatile(
            "{\n"
            "    .reg .pred p;\n"
            "    setp.eq.u32 p, 0, 1;\n"
            "    tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, "
            "{%4,%5,%6,%7}, p;\n"
            "}"
            :
            : "r"(tmem),
              "l"(a0),
              "l"(b0),
              "r"(IDESC),
              "r"(0u),
              "r"(0u),
              "r"(0u),
              "r"(0u)
            : "memory");

        // --------------------------------------------------------
        // MMA #1
        //
        // D += A1 * B1
        // --------------------------------------------------------

        asm volatile(
            "{\n"
            "    .reg .pred p;\n"
            "    setp.eq.u32 p, 1, 1;\n"
            "    tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, "
            "{%4,%5,%6,%7}, p;\n"
            "}"
            :
            : "r"(tmem),
              "l"(a1),
              "l"(b1),
              "r"(IDESC),
              "r"(0u),
              "r"(0u),
              "r"(0u),
              "r"(0u)
            : "memory");

        // --------------------------------------------------------
        // MMA #2
        //
        // D += A2 * B2
        // --------------------------------------------------------

        asm volatile(
            "{\n"
            "    .reg .pred p;\n"
            "    setp.eq.u32 p, 1, 1;\n"
            "    tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, "
            "{%4,%5,%6,%7}, p;\n"
            "}"
            :
            : "r"(tmem),
              "l"(a2),
              "l"(b2),
              "r"(IDESC),
              "r"(0u),
              "r"(0u),
              "r"(0u),
              "r"(0u)
            : "memory");

        // --------------------------------------------------------
        // MMA #3
        //
        // D += A3 * B3
        // --------------------------------------------------------

        asm volatile(
            "{\n"
            "    .reg .pred p;\n"
            "    setp.eq.u32 p, 1, 1;\n"
            "    tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, "
            "{%4,%5,%6,%7}, p;\n"
            "}"
            :
            : "r"(tmem),
              "l"(a3),
              "l"(b3),
              "r"(IDESC),
              "r"(0u),
              "r"(0u),
              "r"(0u),
              "r"(0u)
            : "memory");
    }

    // ============================================================
    // Commit all previously issued tcgen05.mma operations.
    //
    // IMPORTANT:
    //
    // It is .shared::cluster, NOT .shared::cta.
    //
    // This was the source of your previous ptxas error.
    // ============================================================

    if (tid == 0) {
        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one."
            "shared::cluster.b64 [%0];"
            :
            : "r"(mbar_addr)
            : "memory");
    }

    // ============================================================
    // (5) Wait for MMA completion
    // ============================================================

    mbar_wait(mbar_addr, 0);

    // ============================================================
    // Before TMEM -> register loads.
    //
    // tcgen05.mma -> mbarrier wait -> thread synchronization
    // boundary -> tcgen05.ld
    // ============================================================

    asm volatile(
        "tcgen05.fence::after_thread_sync;"
        ::: "memory");

    // ============================================================
    // (6) TMEM -> registers -> global
    //
    // 128 rows are split across 4 warps:
    //
    //   warp 0 -> rows   0..31
    //   warp 1 -> rows  32..63
    //   warp 2 -> rows  64..95
    //   warp 3 -> rows  96..127
    //
    // Each .32x32b.x8 load:
    //
    //   32 lanes
    //   x 8 FP32 values/lane
    //
    // covers 32 rows x 8 columns.
    //
    // N = 64 => 8 loads of 8 columns.
    // ============================================================

    const int row_base = warp * 32;

    for (int n0 = 0; n0 < N; n0 += 8) {

        float x[8];

        // taddr:
        //
        //   high 16 bits = row offset
        //   low  16 bits = column offset
        //
        // For Layout D:
        //   local row = lane
        //   warp selects 32-row partition.
        //
        const uint32_t addr =
            tmem
            + (static_cast<uint32_t>(row_base) << 16)
            + static_cast<uint32_t>(n0);

        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(x[0]),
              "=f"(x[1]),
              "=f"(x[2]),
              "=f"(x[3]),
              "=f"(x[4]),
              "=f"(x[5]),
              "=f"(x[6]),
              "=f"(x[7])
            : "r"(addr)
            : "memory");

        // tcgen05.ld is asynchronous.
        asm volatile(
            "tcgen05.wait::ld.sync.aligned;"
            ::: "memory");

        // Each lane corresponds to one row.
        const int m = row_base + lane;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            gD[m * N + n0 + j] = x[j];
        }
    }

    // ============================================================
    // All TMEM readers must finish before deallocation.
    // ============================================================

    __syncthreads();

    // ============================================================
    // (7) TMEM deallocation
    //
    // dealloc is sync.aligned, so all 32 lanes of warp 0 execute it.
    // ============================================================

    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1."
            "sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem),
              "r"(TMEM_COLS)
            : "memory");
    }
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    tcgen05_tile<<<1, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    printf(bad ? "FAIL seed=%u: %ld / %d\n" : "PASS seed=%u\n", seed,
           bad ? bad : (long)seed, M * N);
    return bad != 0;
}
