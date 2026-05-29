
// Test for generating rotation matrices using MKL's QR decomposition
#include <vector>
#include <gtest/gtest.h>
#include "rotation.h"

TEST(QRTest, GeneratesOrthogonalMatrix) {
    int64_t dim = 5, seed = 123;
    auto rot = make_rotation_matrix(dim, seed);

    // Check if the matrix is orthogonal: Q^T * Q = I
    std::vector<double> identity(dim * dim, 0.0);
    for (int64_t i = 0; i < dim; ++i) {
        identity[i * dim + i] = 1.0;
    }

    std::vector<double> qtq(dim * dim, 0.0);
    // Compute Q^T * Q
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            for (int64_t k = 0; k < dim; ++k) {
                qtq[i * dim + j] += rot[k * dim + i] * rot[k * dim + j];
            }
        }
    }

    // Check if qtq is close to identity
    for (int64_t i = 0; i < dim * dim; ++i) {
        EXPECT_NEAR(qtq[i], identity[i], 1e-6);
    }
}
