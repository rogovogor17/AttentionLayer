#include <benchmark/benchmark.h>

#include "attention.hpp"

static void CreateTensors(Tensor3D<float>& A, Tensor3D<float>& B,
                          Tensor3D<float>& C) {
    int batch = 4, seq_q = 1024, seq_k = 512, d_k = 512, d_v = 1024;
    static Tensor3D<float> A0(batch, seq_q, d_k, -1e6f, 1e6f);
    static Tensor3D<float> B0(batch, seq_k, d_k, -1e6f, 1e6f);
    static Tensor3D<float> C0(batch, seq_k, d_v, -1e6f, 1e6f);
    A = A0;
    B = B0;
    C = C0;
}

static void BM_NaiveMatmul(benchmark::State& state) {
    Tensor3D<float> A, B, C;
    CreateTensors(A, B, C);

    for (auto _ : state) {
        auto result = tensorMul(A, B, NAIVE);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_NaiveMatmul);

static void BM_CachedMatmul(benchmark::State& state) {
    Tensor3D<float> A, B, C;
    CreateTensors(A, B, C);

    for (auto _ : state) {
        auto result = tensorMul(A, B, CACHE_OPTIMIZED);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CachedMatmul);

static void BM_SIMDMatmul(benchmark::State& state) {
    Tensor3D<float> A, B, C;
    CreateTensors(A, B, C);

    for (auto _ : state) {
        auto result = tensorMul(A, B, SIMD);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SIMDMatmul);

static void BM_NaiveAttention(benchmark::State& state) {
    Tensor3D<float> Q, K, V;
    CreateTensors(Q, K, V);

    for (auto _ : state) {
        auto result = attention_with_matmul(Q, K, V, NAIVE);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_NaiveAttention);

static void BM_CachedAttention(benchmark::State& state) {
    Tensor3D<float> Q, K, V;
    CreateTensors(Q, K, V);

    for (auto _ : state) {
        auto result = attention_with_matmul(Q, K, V, CACHE_OPTIMIZED);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CachedAttention);

static void BM_SIMDAttention(benchmark::State& state) {
    Tensor3D<float> Q, K, V;
    CreateTensors(Q, K, V);

    for (auto _ : state) {
        auto result = attention_with_matmul(Q, K, V, SIMD);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SIMDAttention);
