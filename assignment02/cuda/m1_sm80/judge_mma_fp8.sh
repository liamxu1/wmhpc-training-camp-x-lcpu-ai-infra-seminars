#!/usr/bin/env bash
# 问题 1.3 的判测:编译你的 03_mma_fp8.cu,五个 seed 全 PASS 才算过。
# 用法:./judge_mma_fp8.sh 03_mma_fp8.cu
set -e
SRC="${1:?用法: ./judge_mma_fp8.sh <你的.cu>}"
ARCH="${ARCH:-100f}"
BIN=$(mktemp /tmp/mma_fp8.XXXXXX)
nvcc -O2 -std=c++17 -I.. -gencode arch=compute_${ARCH},code=sm_${ARCH} "$SRC" -o "$BIN"
ok=1
for s in 1 7 42 1234 99999; do
    out=$("$BIN" "$s") || ok=0
    echo "$out"
    [[ "$out" == PASS* ]] || ok=0
done
rm -f "$BIN"
[[ $ok == 1 ]] && echo "JUDGE: PASS" || { echo "JUDGE: FAIL"; exit 1; }
