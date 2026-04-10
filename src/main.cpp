#include <iostream>

#include "tensor.hpp"

int main() {
    Tensor<float> tnr;
    tnr.setShape({1, 2, 3});
    tnr.print();
    return 0;
}
