#include <vector>

#include "matrix.hpp"

int main() {
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<float> A(3, 3, values.begin(), values.end());
    A.dump();
    return 0;
}
