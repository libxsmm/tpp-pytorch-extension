#include <iostream>
#include "ext_tpp.h"
#include "timing.h"
#include "xsmm_functors.h"
#include "rotation.h"
using namespace tpp;

int main() {
  
    int64_t dim = 10, seed = 100;
    std::cout << "Hello, TurboQuant!" << std::endl;

    auto rot = make_rotation_matrix(dim, seed);
    std::cout << "Generated rotation matrix:" << std::endl;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            std::cout << rot[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}