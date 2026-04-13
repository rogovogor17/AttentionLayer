#include <cmath>
#include <fstream>
#include <iostream>

#include "matrix.hpp"
#include "tensor3d.hpp"

int main() {
    std::ofstream log("log.log");

    int batch = 8, seq_q = 3, seq_k = 5, d_k = 3, d_v = 9;

    Tensor3D<float> Q(batch, seq_q, d_k, -100.0f, 100.0f);
    Tensor3D<float> K(batch, seq_k, d_k, -100.0f, 100.0f);
    Tensor3D<float> V(batch, seq_k, d_v, -100.0f, 100.0f);
    Q.dump(log);

    K.transpose();
    Tensor3D<float> A = tensorMul(Q, K, NAIVE);
    Tensor3D<float> B = tensorMul(Q, K, CACHE_OPTIMIZED);
    std::cout << (A.isApprox(B) ? "Success!" : "Failed!") << std::endl;

    A *= static_cast<float>(1 / std::sqrt(d_k));
    B *= static_cast<float>(1 / std::sqrt(d_k));
    std::cout << (A.isApprox(B) ? "Success!" : "Failed!") << std::endl;

    return 0;
}
