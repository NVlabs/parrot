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
    auto data = parrot::scalar(9).repeat(30).rand().print();
    data.sort().print().rle().print().max_by_key(parrot::snd()).print();
    // Output:
    // 2 5 6 2 0 0 4 2 3 8 5 0 1 4 4 7 5 8 3 3 8 2 6 0 7 5 6 0 2 3
    // 0 0 0 0 0 1 2 2 2 2 2 3 3 3 3 4 4 4 5 5 5 5 6 6 6 7 7 8 8 8
    // (0, 5) (1, 1) (2, 5) (3, 4) (4, 3) (5, 4) (6, 3) (7, 2) (8, 3)
    // (0, 5)
}
