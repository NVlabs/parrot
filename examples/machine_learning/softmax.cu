#include "parrot.hpp"

using namespace parrot::literals;

auto softmax(auto matrix) {
    auto cols = matrix.shape()[1];
    auto z    = matrix - matrix.maxr(2_ic).replicate(cols);
    auto num  = z.exp();
    auto den  = num.sum(2_ic);
    return num / den.replicate(cols);
}

int main() {
    auto matrix = parrot::range(6).as<float>().reshape({2, 3});
    softmax(matrix).print();
}
