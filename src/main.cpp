#include <vector>

#include "matrix.hpp"

int main() {
    Matrix<float> A = Matrix<float>::eye(3);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<float> B(3, 3, values.begin(), values.end());

    Matrix<float> C = naive_multiply(A, B);
    Matrix<float> D = cached_multiply(A, B);

    return 0;
}
