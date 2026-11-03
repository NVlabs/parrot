/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parrot.hpp"

auto is_in_circle = [] __host__ __device__(float x, float y) {
    // unlike thrust::remove_if, fusion_array::filter 
    // keeps elements that satisfy the predicate
    return x * x + y * y <= 1;
};

int main() {
    int N = 20;

    auto x = parrot::scalar(1.0f).repeat(N).rand();
    auto y = parrot::scalar(1.0f).repeat(N).rand();

    auto p = x.pairs(y) //
                .print()
                .filter(thrust::make_zip_function(is_in_circle));
    
    std::cout << p.size() << std::endl;
    p.print();
}
