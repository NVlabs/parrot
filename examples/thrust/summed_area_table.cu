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
    int m = 3; // number of rows
    int n = 4; // number of columns

    auto data = parrot::matrix(1, {m, n});

    // initial array
    data.print();
    // scan horizontally
    data.sums<1>().print();
    // transpose array
    data.transpose().print();
    // scan transpose horizontally
    data.sums<1>().print();
    // transpose the transpose
    data.transpose().print();

    // alternatively, without the intermidiate prints
    m.sums<2>().sums<1>().print();
}
