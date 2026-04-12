#include <fstream>

#include "tensor3d.hpp"

int main() {
    Tensor3D<float> Q(32, 8, 64);
    Tensor3D<float> K(32, 16, 64);
    Tensor3D<float> V(32, 16, 32);
    std::ofstream log("log.log");
    Q.dump(log);
    return 0;
}
