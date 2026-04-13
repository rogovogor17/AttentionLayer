#include <cmath>
#include <fstream>

#include "attention.hpp"
#include "matrix.hpp"
#include "tensor3d.hpp"

int main() {
    std::ofstream log("log.log");

    int batch = 2, seq_q = 2, seq_k = 1, d_k = 1, d_v = 2;

    Tensor3D<float> Q(batch, seq_q, d_k);
    Q[0][0][0] = -1.1370f;
    Q[0][1][0] = -1.3288f;
    Q[1][0][0] = 0.5047f;
    Q[1][1][0] = 2.3616f;
    Tensor3D<float> K(batch, seq_k, d_k);
    K[0][0][0] = 0.5836f;
    K[1][0][0] = 0.4458f;
    Tensor3D<float> V(batch, seq_k, d_v);
    V[0][0][0] = 0.7172f;
    V[0][0][1] = 0.5812f;
    V[1][0][0] = -0.1118f;
    V[1][0][1] = -0.6680f;

    K.transpose();
    Tensor3D<float> A = tensorMul(Q, K, NAIVE);
    A *= static_cast<float>(1 / std::sqrt(d_k));

    Tensor3D<float> S = softmax(A);
    Tensor3D<float> result = tensorMul(S, V, CACHE_OPTIMIZED);
    result.dump(log);

    return 0;
}
