#include <gtest/gtest.h>

#include "matrix.hpp"

TEST(MatrixMultiplyTest, LargeRandomMatrixTimesIdentity) {
    const int size = 1000;
    Matrix<float> A(size, size, -100.0f, 100.0f);
    Matrix<float> I = Matrix<float>::eye(size);
    Matrix<float> result = cached_multiply(A, I);

    EXPECT_EQ(1, A.isApprox(result));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
