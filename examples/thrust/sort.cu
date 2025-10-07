/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

int main() {
    auto const N = 10;
    auto ints    = parrot::scalar(100).repeat(N).rand();
    auto floats  = parrot::scalar(10.0).repeat(N).rand();

    auto even_first = [] __host__ __device__(int x) { return x % 2; };

    ints.sort().print();                           // sort integers
    ints.sort_by(thrust::greater<int>()).print();  // sort integers descending
    ints.sort_by_key(even_first).print();  // sort integers (user-defined)
    floats.sort().print();                 // sort floats
    floats.pairs(ints).sort().print();     // sort pairs
}
