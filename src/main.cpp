#include <iostream>

#include "matmul.hpp"
#include "tensor.hpp"

int main() {
    Tensor<float> tensor;
    tensor.setShape({3, 3, 3});
    tensor.setData({111, 112, 113, 121, 122, 123, 131, 132, 133,
                    211, 212, 213, 221, 222, 223, 231, 232, 233,
                    311, 312, 313, 321, 322, 323, 331, 332, 333});
    tensor.print();

    Matrix<float> matrix(3, 3, );
    return 0;
}
