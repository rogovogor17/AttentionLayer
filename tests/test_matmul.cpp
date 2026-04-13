#include <gtest/gtest.h>

#include "matrix.hpp"

TEST(MatrixMultiplyTest, LargeMatrix) {
    const int size = 1000;
    Matrix<float> D(size, size, -1e6f, 1e6f);
    Matrix<float> I = Matrix<float>::eye(size);

    Matrix<float> A = cached_multiply(D, I);
    EXPECT_EQ(1, A.isApprox(D));

    Matrix<float> B = naive_multiply(D, I);
    EXPECT_EQ(1, B.isApprox(D));

    Matrix<float> C = simd_multiply(D, I);
    EXPECT_EQ(1, C.isApprox(D));

    EXPECT_EQ(1, A.isApprox(B));
    EXPECT_EQ(1, B.isApprox(C));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
