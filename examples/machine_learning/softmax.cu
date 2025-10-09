#include "parrot.hpp"

auto softmax(auto matrix) {
    auto cols = matrix.shape()[1];
    auto z    = matrix - matrix.template maxr<2>().replicate(cols);
    auto num  = z.exp();
    auto den  = num.template sum<2>();
    return num / den.replicate(cols);
}

int main() {
    auto matrix = parrot::range(6).as<float>().reshape({2, 3});
    softmax(matrix).print();
}
