
// Random orthogonal matrix generation using QR decomposition
// Using MKL LAPACKE for QR decomposition

#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>

// include mkl headers for QR decomposition
#include <mkl.h>


std::vector<double> make_rotation_matrix(int64_t dim, int64_t seed) {
    std::mt19937 rng(seed); // Fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0f, 1.0f);

    // Create a random matrix
    std::vector<double> mat(dim * dim);
    std::generate(mat.begin(), mat.end(), [&]() { return dist(rng); });
    
    // Perform QR decomposition to get an orthogonal matrix
    std::vector<double> tau(dim); // For storing the scalar factors of the elementary reflectors
    
    auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim, dim, mat.data(), dim, tau.data());

    if (info != 0) {
        throw std::runtime_error("QR decomposition failed");
    }

    // Get r matrix from the upper triangle of the decomposed matrix
    // Used form sign correction later
    std::vector<double> r(dim * dim, 0.0);
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = i; j < dim; ++j) {
            r[i * dim + j] = mat[i * dim + j];
        }
    }

    // Generate Q from the Householder reflectors
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dim, dim, dim, mat.data(), dim, tau.data());
    if (info != 0) {
        throw std::runtime_error("Generating Q from QR decomposition failed");
    }

    // Sign correction: Q = Q * diag(sign(diag(R)))
    for (int64_t i = 0; i < dim; ++i) {
        if (r[i * dim + i] < 0) {
            for (int64_t j = 0; j < dim; ++j) {
                mat[i * dim + j] = -mat[i * dim + j];
            }
        }
    }

    return mat; // Return the orthogonal matrix as the rotation matrix
}