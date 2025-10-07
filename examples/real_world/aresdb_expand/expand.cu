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

// https://github.com/uber/aresdb/blob/a8d2aedc6850b10a6cc9381ba780800290b2756d/query/sort_reduce.cu#L252
// to 314

// this is a simplified version of the expand
// function that assumes all dim widths are 1
template <typename BaseCountsArray,
          typename IndexArray,
          typename InputKeysArray>
auto expand_parrot(const InputKeysArray &input_keys,
                   const BaseCountsArray &base_counts,
                   const IndexArray &indices,
                   int capacity) {
    auto counts = base_counts.deltas()
                    .gather(indices)
                    .sums()
                    .min(capacity)
                    .prepend(0)
                    .deltas()
                    .cycle({input_keys.size()});
    return input_keys.replicate(counts);
}
