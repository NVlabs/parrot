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

int main() {
    parrot::array H({14, 20, 38, 46});
    
    std::cout << H.size() << std::endl; // 4
    H.print();  // 14 20 38 46

    H = H.take(2);
    std::cout << H.size() << std::endl; // 2
    H.print();  // 14 20

    parrot::array D = H;

    // TODO: not possible in parrot?
    D[0] = 99;
    D[1] = 88;

    D.print();
}
