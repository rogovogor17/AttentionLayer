#include <fstream>

#include "tensor3d.hpp"

int main() {
    std::ofstream log("log.log");

    Tensor3D<float> Q(32, 8, 64, -100.0f, 100.0f);
    Tensor3D<float> K(32, 16, 64, -100.0f, 100.0f);
    Tensor3D<float> V(32, 16, 32, -100.0f, 100.0f);

    Q.dump(log);

    return 0;
}
