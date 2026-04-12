#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "matrix.hpp"

TEST(MatrixMultiplyTest, LargeRandomMatrixTimesIdentity) {
    const int size = 1000;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0, 100.0);

    Matrix<float> A(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[i][j] = dist(gen);
        }
    }

    Matrix<float> I = Matrix<float>::eye(size);
    Matrix<float> result = cached_multiply(A, I);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            EXPECT_FLOAT_EQ(result[i][j], A[i][j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
