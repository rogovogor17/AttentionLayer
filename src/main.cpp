#include <fstream>

#include "attention.hpp"
#include "matrix.hpp"
#include "tensor3d.hpp"

/*TODO:
1. Online softmax
2. SIMD
3. Readme
4. Gitlab
*/

int main() {
    std::ofstream log("log.log");

    int batch = 2, seq_q = 2, seq_k = 1, d_k = 1, d_v = 2;

    Tensor3D<float> Q(batch, seq_q, d_k, -2.0f, 2.0f);
    Q.dump(log);
    Tensor3D<float> K(batch, seq_k, d_k, -2.0f, 2.0f);
    K.dump(log);
    Tensor3D<float> V(batch, seq_k, d_v, -2.0f, 2.0f);
    V.dump(log);

    Tensor3D<float> result = attention_with_matmul(Q, K, V, CACHE_OPTIMIZED);

    return 0;
}
