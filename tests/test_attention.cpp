#include <gtest/gtest.h>

#include "attention.hpp"

TEST(AttentionTest, OnlineSoftmaxCheck) {
    int batch = 4, seq_q = 1024, seq_k = 512, d_k = 512, d_v = 1024;
    Tensor3D<float> Q(batch, seq_q, d_k, -1e6f, 1e6f);
    Tensor3D<float> K(batch, seq_k, d_k, -1e6f, 1e6f);
    Tensor3D<float> V(batch, seq_k, d_v, -1e6f, 1e6f);

    Tensor3D<float> A = attention_online(Q, K, V);
    Tensor3D<float> B = attention_with_matmul(Q, K, V, CACHE_OPTIMIZED);

    EXPECT_EQ(1, A.isApprox(B));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
