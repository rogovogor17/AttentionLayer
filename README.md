<a id="readme-top"></a>

[![C++][cpp-shield]][cpp-url]
[![CMake][cmake-shield]][cmake-url]
[![Benchmark][benchmark-shield]][benchmark-url]
[![Tests][tests-shield]][tests-url]
[![SIMD][simd-shield]][simd-url]

<br />
<div align="center">
  <a href="https://github.com/your_username/attention">
    <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/CPP.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Scaled Dot-Product Attention</h3>
  <h4 align="center">High-Performance C++ Implementation with Optimized Matrix Multiplication</h4>

<div align="center">

_["Attention Is All You Need" — Vaswani et al., NeurIPS 2017](./docs/Attention_is_all_you_need.pdf)_

**[View Benchmark Results](#-benchmark-results)**

[Getting Started](#-getting-started) • [Architecture](#-architecture) • [Implemented Methods](#-implemented-methods) • [Performance Analysis](#-performance-analysis)

</div>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-about-the-project">About The Project</a>
      <ul>
        <li><a href="#mathematical-formulation">Mathematical Formulation</a></li>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#-built-with">Built With</a></li>
	<li><a href="#benchmarked-on">Benchmarked On</a></li>
    <li><a href="#-getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#building">Building</a></li>
      </ul>
    </li>
    <li><a href="#-architecture">Architecture</a>
      <ul>
        <li><a href="#class-hierarchy">Class Hierarchy</a></li>
        <li><a href="#module-structure">Module Structure</a></li>
      </ul>
    </li>
    <li><a href="#-implemented-methods">Implemented Methods</a>
      <ul>
        <li><a href="#matrix-multiplication-algorithms">Matrix Multiplication Algorithms</a></li>
        <li><a href="#attention-mechanisms">Attention Mechanisms</a></li>
      </ul>
    </li>
    <li><a href="#-benchmark-results">Benchmark Results</a>
      <ul>
        <li><a href="#matrix-multiplication-performance">Matrix Multiplication Performance</a></li>
        <li><a href="#attention-performance">Attention Performance</a></li>
        <li><a href="#speedup-summary">Speedup Summary</a></li>
      </ul>
    </li>
    <li><a href="#-testing">Testing</a></li>
    <li><a href="#-performance-analysis">Performance Analysis</a>
      <ul>
        <li><a href="#why-cache-optimization-matters">Why Cache Optimization Matters</a></li>
        <li><a href="#why-simd-matters">Why SIMD Matters</a></li>
        <li><a href="#complexity-comparison">Complexity Comparison</a></li>
      </ul>
    </li>
    <li><a href="#-roadmap">Roadmap</a></li>
    <li><a href="#-references">References</a></li>
    <li><a href="#-contact">Contact</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## About The Project

This project implements the **Scaled Dot-Product Attention** mechanism in **C++17/20** with a focus on computational efficiency. Three distinct matrix multiplication implementations are provided, ranging from naive versions to SIMD-optimized code.

### Mathematical Formulation

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" alt="Attention Formula">
</div>

<br />

| Tensor          | Dimensions            | Description                     |
| --------------- | --------------------- | ------------------------------- |
| **Q** (Queries) | `[batch, seq_q, d_k]` | Query vectors for each position |
| **K** (Keys)    | `[batch, seq_k, d_k]` | Key vectors for each position   |
| **V** (Values)  | `[batch, seq_k, d_v]` | Value vectors for each position |
| **Output**      | `[batch, seq_q, d_v]` | Context-aware representations   |

### Key Features

| Feature                    | Status   | Description                                |
| -------------------------- | -------- | ------------------------------------------ |
| **Naive MatMul**           | Complete | Classic triple-loop O(n³) implementation   |
| **Cache-Optimized MatMul** | Complete | Blocked algorithm with 32×32 tiling        |
| **SIMD-Vectorized MatMul** | Complete | AVX2/AVX intrinsics (8 floats / 4 doubles) |
| **Standard Attention**     | Complete | Full matrix materialization with softmax   |
| **Online Softmax**         | Complete | FlashAttention-like memory optimization    |
| **Benchmarks**             | Complete | Google Benchmark integration               |
| **Unit Testing**           | Complete | Google Test suite                          |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Built With

<table align="center">
  <tr>
    <td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cplusplus/cplusplus-original.svg" width="50"><br><b>C++17/20</b></td>
    <td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cmake/cmake-original.svg" width="50"><br><b>CMake 3.20+</b></td>
    <td align="center"><img src="https://www.doxygen.nl/images/doxygen.png" width="100"><br><b>Doxygen</b></td>
    <td align="center"><img src="https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg" width="40"><br><b>Google Benchmark</b></td>
	<td align="center"><img src="https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg" width="40"><br><b>Google Test</b></td>
  </tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Benchmarked On

<table>
  <tr>
    <th>Component</th>
    <th>Specification</th>
  </tr>
  <tr>
    <td><b>CPU</b></td>
    <td>12th Gen Intel(R) Core(TM) i5-12600KF<br></td>
  </tr>
  <tr>
    <td><b>Architecture</b></td>
    <td>x86_64, 64-bit, Little-endian<br><i>NUMA (1 socket)</i></td>
  </tr>
  <tr>
    <td><b>Cache Hierarchy</b></td>
    <td>L1: 42KB (data) + 45KB (inst) per core<br>L2: 1.4MB per core<br>L3: 20MB</td>
  </tr>
  <tr>
    <td><b>Memory</b></td>
    <td>DDR4-2400 16GB<br></td>
  </tr>
  <tr>
    <td><b>SIMD Extensions</b></td>
    <td>AVX2 (256-bit), SSE4.2 (128-bit), FMA</td>
  </tr>
  <tr>
    <td><b>OS & Compiler</b></td>
    <td>Ubuntu 24.04<br>GCC 13.3+, Clang 18+, with <code>-march=native -O3 -DNDEBUG</code></td>
  </tr>
</table>

---

## Getting Started

### Prerequisites

| Tool              | Version       | Installation Command (Ubuntu)   |
| ----------------- | ------------- | ------------------------------- |
| **CMake**         | ≥ 3.20        | `sudo apt install cmake`        |
| **C++ Compiler**  | C++17 capable | GCC 9+, Clang 12+               |
| **Git**           | Latest        | `sudo apt install git`          |
| **.clang-format** | 22.0.0        | `sudo apt install clang-format` |
| **Doxygen**       | 1.16.0        | `sudo apt install doxygen`      |

### Installation

```bash
# Clone the repository
git clone https://github.com/rogovogor17/AttentionLayer
cd AttentionLayer
```

### Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build everything
cmake --build .
```

### Running

```bash
# Run matrix multiplication benchmarks
./benchmarks/bench_matmul

# Run attention benchmarks
./benchmarks/bench_softmax

# Run all tests with CTest
ctest
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Architecture

### Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tensor3D<T>                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 3D Tensor as Batch of Matrices                            │  │
│  │ • Batch dimension for parallel sequence processing        │  │
│  │ • Row/Column dimensions for each matrix                   │  │
│  │ • Vector storage of Matrix<T> objects                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                        Matrix<T>                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2D Matrix with Row-Major Layout                           │  │
│  │ • ProxyRow for [][] bracket access                        │  │
│  │ • Random initialization (uniform distribution)            │  │
│  │ • Transpose (in-place & const)                            │  │
│  │ • Equality with epsilon tolerance                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    std::vector<T> data_                         │
│              (Contiguous memory for cache locality)             │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

| File                 | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `matrix.hpp`         | Matrix class + 3 multiplication algorithms (naive, cached, SIMD) |
| `tensor3d.hpp`       | 3D tensor wrapper with batch operations                          |
| `attention.hpp`      | Attention mechanism + online softmax optimization                |
| `bench_matmul.cpp`   | Matrix multiplication benchmarks                                 |
| `bench_softmax.cpp`  | Attention vs online softmax benchmarks                           |
| `test_matmul.cpp`    | Matrix multiplication unit tests                                 |
| `test_attention.cpp` | Attention correctness tests                                      |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Implemented Methods

### Matrix Multiplication Algorithms

#### 1. Naive Multiplication (`naive_multiply`)

The classic triple-loop implementation.

```cpp
for (int i = 0; i < M; i++)
	for (int j = 0; j < N; j++)
		for (int k = 0; k < K; k++)
			C[i][j] += A[i][k] * B[k][j];
```

**Complexity:** O(M·N·K) — no optimizations.

#### 2. Cache-Optimized Multiplication (`cached_multiply`)

Blocked algorithm exploiting temporal locality via loop tiling.

```cpp
const int bs = 32;  // Fits in L1/L2 cache
for (int ii = 0; ii < M; ii += bs)
    for (int jj = 0; jj < N; jj += bs)
        for (int kk = 0; kk < K; kk += bs)
            for (int i = ii; i < std::min(ii + bs, M); i++)
                for (int k = kk; k < std::min(kk + bs, K); k++) {
                    T A_ik = A[i][k];
                    for (int j = jj; j < std::min(jj + bs, N); j++)
                        C[i][j] += A_ik * B[j][k];  // B^T for sequential access
                }
```

**Optimizations:**

- Block tiling (32×32)
- B matrix pre-transposed
- Sequential memory access patterns

#### 3. SIMD-Vectorized Multiplication (`simd_multiply`)

AVX2 intrinsics for 8-way (float) or 4-way (double) parallelism.

```cpp
#if X86_SIMD_AVAILABLE
using traits = simd_traits<T>;
constexpr int simd_size = traits::size;  // 8 for float, 4 for double

for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
        auto a = traits::set1(A[i][k]);        // Broadcast
        for (int j = 0; j < N; j += simd_size) {
            auto b = traits::load(&B[k][j]);   // Load SIMD vector
            auto c = traits::load(&C[i][j]);   // Load current
            c = traits::fmadd(a, b, c);        // FMA: c = a * b + c
            traits::store(&C[i][j], c);        // Store back
        }
    }
#endif
```

**SIMD Capabilities:**

- **FMA instructions** (Fused Multiply-Add)
- **8-way float32** or **4-way float64** parallelism
- **Fallback to cache-optimized** when SIMD unavailable

### Attention Mechanisms

#### Standard Attention (`attention_with_matmul`)

```cpp
K.transpose();									 //matmul_type:
Tensor3D<T> A = tensorMul(Q, K, matmul_type);    //NAIVE
A *= static_cast<T>(1.0 / std::sqrt(Q.ncols())); //CACHED_OPTIMIZED
return tensorMul(softmax(A), V, matmul_type);	 //SIMD
```

**Memory:** O(seq_q × seq_k) — full attention matrix materialized.

#### Online Softmax (FlashAttention-like) (`attention_online`)

```cpp
Tensor3D<T> result(batch, seq_q, d_v);
T scale = static_cast<T>(1.0 / std::sqrt(d_k));
for (int b = 0; b < batch; b++) {
	for (int i = 0; i < seq_q; i++) {
		T sum = T{};
		std::vector<T> row(d_v, T{});		   //Current transformed row in tensor
		const T* q_row = &Q[b][i][0];
		for (int k = 0; k < seq_k; k++) {
			T score = 0;
			const T* k_row = &K[b][k][0];
			for (int j = 0; j < d_k; j++)
				score += q_row[j] * k_row[j];  //First Matrix multiplication
			score = std::exp(score * scale);   //Softmax transformation
			sum += score;
			const T* v_row = &V[b][k][0];
			for (int j = 0; j < d_v; j++)
				row[j] += score * v_row[j];	   //Second Matrix multiplication
		}
		T* result_row = &result[b][i][0];
		for (int j = 0; j < d_v; j++)
			result_row[j] = row[j] / sum;	   //Softmax normalization
	}
}
```

**Memory:** O(d_v) — no attention matrix materialized!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Testing

### Test Coverage

| Test Suite                | Tests                                | Status |
| ------------------------- | ------------------------------------ | ------ |
| **Matrix Multiplication** | Identity preservation (1000×1000)    | Pass   |
|                           | Naive == Cached == SIMD              | Pass   |
| **Attention**             | Online softmax == Standard attention | Pass   |
|                           | Tolerance-based float comparison     | Pass   |

### Running Tests

```bash
cd build && ctest --output-on-failure
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Benchmark Results

### Test Configuration

| Parameter                     | Value                             |
| ----------------------------- | --------------------------------- |
| **Batch Size**                | 4                                 |
| **Sequence Length (Q)**       | 1024                              |
| **Sequence Length (K/V)**     | 512                               |
| **Embedding Dimension (d_k)** | 512                               |
| **Value Dimension (d_v)**     | 1024                              |
| **Data Type**                 | `float` (32-bit)                  |
| **Hardware**                  | x86-64 with AVX2                  |
| **Compiler**                  | GCC 13.3 with `-O3 -march=native` |

### Matrix Multiplication Performance

### Attention Performance

### Speedup Summary

| Operation     | Naive → Cache | Cache → SIMD | Naive → SIMD | Online vs Cache |
| ------------- | :-----------: | :----------: | :----------: | :-------------: |
| **MatMul**    |     2.81×     |    2.07×     |    5.83×     |        —        |
| **Attention** |     2.80×     |    2.06×     |    5.78×     |      2.65×      |

<div align="center">
  <img src="https://quickchart.io/chart?c={type:'bar',data:{labels:['Naive','Cache','SIMD','Online'],datasets:[{label:'Time (ms)',data:[732.5,261.3,126.7,98.4],backgroundColor:['#ef4444','#f59e0b','#10b981','#06b6d4']}]},options:{title:{display:true,text:'Attention Performance Comparison'}}}" alt="Performance Chart">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Performance Analysis

### Why Cache Optimization Matters

Matrix multiplication's natural access pattern defeats simple caching:

```
Naive: C[i][j] += A[i][k] * B[k][j]
                ↑           ↑
            sequential    strided (bad!)
```

**Blocked algorithm benefits:**

- 32×32 block fits in L1 cache (32×32×4 bytes = 4KB per matrix)
- 8× fewer cache misses vs naive
- ~3× speedup on large matrices

### Why SIMD Matters

Modern CPUs can process multiple operations in one cycle:

| Extension | Floats per instruction | FMA support |
| --------- | :--------------------: | :---------: |
| SSE       |           4            |     No      |
| AVX       |           8            |     Yes     |

**FMA (Fused Multiply-Add):** `c = a * b + c` in **1 cycle** instead of 2!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] **Phase 1: Foundation**
  - [x] Matrix class with row-major storage
  - [x] Naive multiplication
  - [x] Cache-optimized multiplication

- [x] **Phase 2: SIMD Optimization**
  - [x] AVX2 intrinsics for float/double
  - [x] FMA instruction support
  - [x] Fallback for non-SIMD architectures

- [x] **Phase 3: Attention**
  - [x] Tensor3D batch processing
  - [x] Standard attention with softmax
  - [x] Online softmax (FlashAttention-like)

- [x] **Phase 4: Testing & Benchmarking**
  - [x] Google Benchmark integration
  - [x] Google Test suite
  - [x] Performance analysis

- [ ] **Phase 5: Future Enhancements**
  - [ ] Multi-threading (OpenMP/TBB)
  - [ ] ARM NEON support (Apple Silicon)
  - [ ] Multi-Head Attention
  - [ ] FlashAttention v2 kernel fusion

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## References

1. **Attention Is All You Need** — Vaswani, A., et al. _NeurIPS 2017_
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **FlashAttention: Fast and Memory-Efficient Exact Attention** — Dao, T., et al. _arXiv 2022_
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

3. **Intel Intrinsics Guide** — [Intel.com](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<div align="center">
  <sub>© 2026 — High-Performance Attention in C++</sub>
</div>

[cpp-shield]: https://img.shields.io/badge/C++-17/20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://isocpp.org/
[cmake-shield]: https://img.shields.io/badge/CMake-3.20+-064F8C?style=for-the-badge&logo=cmake&logoColor=white
[cmake-url]: https://cmake.org/
[benchmark-shield]: https://img.shields.io/badge/Benchmark-Google-4285F4?style=for-the-badge&logo=google&logoColor=white
[benchmark-url]: https://github.com/google/benchmark
[tests-shield]: https://img.shields.io/badge/Tests-Google_Test-4285F4?style=for-the-badge&logo=google&logoColor=white
[tests-url]: https://github.com/google/googletest
[license-shield]: https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge
[license-url]: LICENSE
[simd-shield]: https://img.shields.io/badge/SIMD-AVX2-0078D4?style=for-the-badge&logo=intel&logoColor=white
[simd-url]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
