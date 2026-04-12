#include <vector>

#include "matrix.hpp"

int main() {
    std::vector<float> values = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<float> A(3, 3, values.begin(), values.end());
    Matrix<float> B(3, 3, 5);
    Matrix<float> C = naive_multiply(A, B);
    C.dump();
    return 0;
}
