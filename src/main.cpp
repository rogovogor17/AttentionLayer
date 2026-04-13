#include <fstream>

#include "matrix.hpp"
#include "tensor3d.hpp"

int main() {
    std::ofstream log("log.log");

    Tensor3D<float> Q(8, 3, 3, -100.0f, 100.0f);
    Tensor3D<float> I = Tensor3D<float>::eye(8, 3);
    Q.dump(log);

    Tensor3D<float> A = tensorMul(Q, I, NAIVE);
    A.dump(log);

    Tensor3D<float> B = tensorMul(Q, I, CACHE_OPTIMIZED);
    B.dump(log);

    return 0;
}
