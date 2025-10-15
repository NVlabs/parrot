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

#include <sstream>
#include <stdexcept>
#include <string>
#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

// Test the stats::norm_cdf function
TEST_CASE("ParrotTest - StatsFunctions") {
    SUBCASE("norm_cdf function") {
        auto arr    = parrot::array<float>({0.0F, 1.0F, -1.0F});
        auto result = parrot::stats::norm_cdf(arr);

        // Expected values from standard normal CDF:
        // norm_cdf(0) = 0.5
        // norm_cdf(1) ≈ 0.8413
        // norm_cdf(-1) ≈ 0.1587
        auto expected = parrot::array<float>({0.5F, 0.8413447F, 0.1586553F});

        CHECK(result.size() == 3);
        auto result_host = result.to_host();

        CHECK(doctest::Approx(result_host[0]) == expected.to_host()[0]);
        CHECK(doctest::Approx(result_host[1]).epsilon(0.0001) ==
              expected.to_host()[1]);
        CHECK(doctest::Approx(result_host[2]).epsilon(0.0001) ==
              expected.to_host()[2]);
    }
}

// Test keep function with a range and stencil as in the example
TEST_CASE("ParrotTest - KeepRangeStencilTest") {
    auto stencil = parrot::array({0, 1, 1, 0, 0, 1, 0, 1});
    // parrot::range(8) -> [1, 2, 3, 4, 5, 6, 7, 8]
    // .minus(1) -> [0, 1, 2, 3, 4, 5, 6, 7]
    // .keep(stencil) -> keep elements where stencil is 1 -> indices [1, 2, 5,
    // 7]
    auto indices  = parrot::range(8).minus(1).keep(stencil);
    auto expected = parrot::array({1, 2, 5, 7});
    CHECK(check_match(indices, expected));
}

// Test the shape functionality
TEST_CASE("ParrotTest - ShapeTest") {
    // Test shape of range
    auto arr1 = parrot::range(10);
    REQUIRE_EQ(arr1.shape().size(), 1);
    CHECK_EQ(arr1.shape()[0], 10);

    // Test shape after operations
    auto arr2 = arr1.times(2);
    REQUIRE_EQ(arr2.shape().size(), 1);
    CHECK_EQ(arr2.shape()[0], 10);

    // Test shape after append
    auto arr3 = arr1.append(11);
    REQUIRE_EQ(arr3.shape().size(), 1);
    CHECK_EQ(arr3.shape()[0], 11);

    // Test shape after prepend
    auto arr4 = arr1.prepend(0);
    REQUIRE_EQ(arr4.shape().size(), 1);
    CHECK_EQ(arr4.shape()[0], 11);

    // Test shape after take
    auto arr5 = arr1.take(5);
    REQUIRE_EQ(arr5.shape().size(), 1);
    CHECK_EQ(arr5.shape()[0], 5);

    // Test shape after filtering
    auto stencil = parrot::array({1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
    auto arr6    = arr1.keep(stencil);
    REQUIRE_EQ(arr6.shape().size(), 1);
    CHECK_EQ(arr6.shape()[0], 5);  // 5 elements kept
}

// Test reshape function (same size)
TEST_CASE("ParrotTest - ReshapeTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6});

    // Reshape to 2x3 (same total size)
    auto reshaped = arr.reshape({2, 3});

    // Check that the shape is correct
    auto shape = reshaped.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);
    CHECK_EQ(shape[1], 3);

    // Verify the total size remains the same
    CHECK_EQ(reshaped.size(), 6);

    // Original array should be unchanged
    REQUIRE_EQ(arr.shape().size(), 1);
    CHECK_EQ(arr.shape()[0], 6);
    CHECK_EQ(arr.size(), 6);

    // Check content
    auto expected = parrot::array({1, 2, 3, 4, 5, 6}).reshape({2, 3});
    CHECK(check_match(reshaped, expected));
}

// Test reshape function with truncation (smaller total size)
TEST_CASE("ParrotTest - ReshapeTruncateTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6});

    // Reshape to 2x2 (smaller total size, truncates to {1, 2, 3, 4})
    auto reshaped = arr.reshape({2, 2});

    // Check that the shape is correct
    auto shape = reshaped.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);
    CHECK_EQ(shape[1], 2);

    // Verify the total size is updated
    CHECK_EQ(reshaped.size(), 4);

    // Original array should be unchanged
    REQUIRE_EQ(arr.shape().size(), 1);
    CHECK_EQ(arr.shape()[0], 6);
    CHECK_EQ(arr.size(), 6);

    // Check content
    auto expected = parrot::array({1, 2, 3, 4});
    CHECK(check_match(reshaped, expected));

    // Check print output
    std::stringstream ss{};
    reshaped.print(ss);
    std::string const expected_print = "1 2\n3 4\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test reshape function with larger shape (should throw exception)
TEST_CASE("ParrotTest - ReshapeLargerSizeTest") {
    auto arr = parrot::array({1, 2, 3});
    // Reshape to 2x3 (larger total size 6 > 3)
    CHECK_THROWS_AS(static_cast<void>(arr.reshape({2, 3})),
                    std::invalid_argument);
}

// Test cycle function with data cycling (larger total size)
TEST_CASE("ParrotTest - CycleLargerSizeTest") {
    auto arr = parrot::array({1, 2, 3});

    // Cycle to 2x3 (larger total size 6 > 3, cycles data to {1, 2, 3, 1, 2, 3})
    auto cycled = arr.cycle({2, 3});

    // Check that the shape is correct
    auto shape = cycled.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);
    CHECK_EQ(shape[1], 3);

    // Verify the total size is updated
    CHECK_EQ(cycled.size(), 6);

    // Original array should be unchanged
    REQUIRE_EQ(arr.shape().size(), 1);
    CHECK_EQ(arr.shape()[0], 3);
    CHECK_EQ(arr.size(), 3);

    // Check content
    auto expected = parrot::array({1, 2, 3, 1, 2, 3});
    CHECK(check_match(cycled, expected));

    // Check print output
    std::stringstream ss{};
    cycled.print(ss);
    std::string const expected_print = "1 2 3\n1 2 3\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test cycle with equal size (behaves like reshape)
TEST_CASE("ParrotTest - CycleEqualSizeTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6});

    // Cycle to 2x3 (same total size as original)
    auto cycled = arr.cycle({2, 3});

    // Check that the shape is correct
    auto shape = cycled.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);
    CHECK_EQ(shape[1], 3);

    // Verify the total size remains the same
    CHECK_EQ(cycled.size(), 6);

    // Original array should be unchanged
    REQUIRE_EQ(arr.shape().size(), 1);
    CHECK_EQ(arr.shape()[0], 6);
    CHECK_EQ(arr.size(), 6);

    // Check content
    CHECK(check_match(cycled, arr));  // Should contain the same data

    // Check print output
    std::stringstream ss{};
    cycled.print(ss);
    std::string const expected_print = "1 2 3\n4 5 6\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test cycle with smaller size (truncates like reshape)
TEST_CASE("ParrotTest - CycleSmallerSizeTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6});

    // Cycle to 2x2 (smaller total size 4 < 6)
    auto cycled = arr.cycle({2, 2});

    // Check that the shape is correct
    auto shape = cycled.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);
    CHECK_EQ(shape[1], 2);

    // Verify the total size is updated
    CHECK_EQ(cycled.size(), 4);

    // Check content
    auto expected = parrot::array({1, 2, 3, 4});
    CHECK(check_match(cycled, expected));

    // Check print output
    std::stringstream ss{};
    cycled.print(ss);
    std::string const expected_print = "1 2\n3 4\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test multidimensional printing
TEST_CASE("ParrotTest - MultidimensionalPrintTest") {
    auto arr      = parrot::array({1, 2, 3, 4, 5, 6});
    auto reshaped = arr.reshape({2, 3});

    std::stringstream ss{};
    reshaped.print(ss);
    std::string const expected = "1 2 3\n4 5 6\n";
    CHECK_EQ(ss.str(), expected);

    // Test 3D printing
    auto arr3d_src = parrot::array({1, 2, 3, 4, 5, 6, 7, 8});
    auto arr3d     = arr3d_src.reshape({2, 2, 2});
    std::stringstream ss3d{};
    arr3d.print(ss3d);
    // Expected: Layer 0
    // 1 2 3 4 Layer 1 5 6 7 8(or similar)
    // Just check it doesn't crash and produces output for now
    CHECK(!ss3d.str().empty());
}

// Test rank method
TEST_CASE("ParrotTest - RankTest") {
    // Test rank of a 1D array
    auto arr1 = parrot::array({1, 2, 3, 4});
    CHECK_EQ(arr1.rank(), 1);

    // Test rank of a 2D array
    auto arr2 = arr1.reshape({2, 2});
    CHECK_EQ(arr2.rank(), 2);

    // Test rank of a 3D array
    auto arr3  = parrot::array({1, 2, 3, 4, 5, 6, 7, 8});
    auto arr3d = arr3.reshape({2, 2, 2});
    CHECK_EQ(arr3d.rank(), 3);

    // Test rank of a scalar
    auto scalar = parrot::scalar(42);
    CHECK_EQ(scalar.rank(), 0);
}

// Test scalar constructor
TEST_CASE("ParrotTest - ScalarConstructorTest") {
    // Create a scalar array with value 42
    auto scalar = parrot::scalar(42);
    CHECK_EQ(scalar.rank(), 0);
    CHECK_EQ(scalar.value(), 42);
    CHECK_EQ(scalar.size(), 1);  // Size is 1 for scalar

    // Test the factory function for scalars
    auto scalar2 = parrot::scalar(99);
    CHECK_EQ(scalar2.rank(), 0);
    CHECK_EQ(scalar2.value(), 99);
    CHECK_EQ(scalar2.size(), 1);
}

// Test repeat method
TEST_CASE("ParrotTest - RepeatTest") {
    auto scalar = parrot::scalar(7);

    // Repeat to create a 6-element array
    auto repeated = scalar.repeat(6);
    CHECK_EQ(repeated.size(), 6);

    // Check that the shape is correct (1D)
    auto shape = repeated.shape();
    REQUIRE_EQ(shape.size(), 1);
    CHECK_EQ(shape[0], 6);

    // Check content
    auto expected = parrot::array({7, 7, 7, 7, 7, 7});
    CHECK(check_match(repeated, expected));

    // Check print output
    std::stringstream ss{};
    repeated.print(ss);
    std::string const expected_print = "7 7 7 7 7 7\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test repeat method with invalid input
TEST_CASE("ParrotTest - RepeatInvalidTest") {
    // Create a non-scalar array
    auto arr = parrot::array({1, 2, 3});
    CHECK_THROWS_AS(static_cast<void>(arr.repeat(5)),
                    std::invalid_argument);  // Should throw exception

    // Create a scalar array
    auto scalar = parrot::scalar(7);
    CHECK_THROWS_AS(static_cast<void>(scalar.repeat(0)),
                    std::invalid_argument);  // Should throw for n=0
    CHECK_THROWS_AS(static_cast<void>(scalar.repeat(-1)),
                    std::invalid_argument);  // Should throw for n<0
}

// Test matrix function
TEST_CASE("ParrotTest - MatrixTest") {
    // Create a matrix with value 7 and shape {3, 4}
    auto mat = parrot::matrix(7, {3, 4});
    CHECK_EQ(mat.size(), 12);

    // Check that the shape is correct (2D)
    auto shape = mat.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 3);
    CHECK_EQ(shape[1], 4);

    // Check content - all elements should be 7
    auto host_vals = mat.to_host();
    REQUIRE_EQ(host_vals.size(), 12);
    for (int const val : host_vals) { CHECK_EQ(val, 7); }

    // Check print output
    std::stringstream ss{};
    mat.print(ss);
    std::string const expected_print = "7 7 7 7\n7 7 7 7\n7 7 7 7\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test matrix function with invalid inputs
TEST_CASE("ParrotTest - MatrixInvalidTest") {
    CHECK_THROWS_AS(parrot::matrix(5, {10}),
                    std::invalid_argument);  // Shape must have > 1 dimension
    CHECK_THROWS_AS(parrot::matrix(5, {2, 3, 4}),
                    std::invalid_argument);  // Shape must have <= 2 dimensions
    CHECK_THROWS_AS(parrot::matrix(5, {}),
                    std::invalid_argument);  // Shape cannot be empty
}

// Test nested initializer list matrix function
TEST_CASE("ParrotTest - NestedMatrixTest") {
    // Create a 2x3 matrix with integers
    auto mat = parrot::matrix({{1, 2, 3}, {4, 5, 6}});

    // Check size and shape
    CHECK_EQ(mat.size(), 6);
    auto shape = mat.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 2);  // rows
    CHECK_EQ(shape[1], 3);  // cols

    // Check content (row-major order)
    auto expected = parrot::array({1, 2, 3, 4, 5, 6});
    CHECK(check_match(mat, expected));

    // Check print output
    std::stringstream ss{};
    mat.print(ss);
    std::string const expected_print = "1 2 3\n4 5 6\n";
    CHECK_EQ(ss.str(), expected_print);
}

// Test nested initializer list matrix with doubles
TEST_CASE("ParrotTest - NestedMatrixDoubleTest") {
    // Create a 3x2 matrix with doubles
    auto mat = parrot::matrix({{1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}});

    // Check size and shape
    CHECK_EQ(mat.size(), 6);
    auto shape = mat.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 3);  // rows
    CHECK_EQ(shape[1], 2);  // cols

    // Check content
    auto expected = parrot::array({1.5, 2.5, 3.5, 4.5, 5.5, 6.5});
    CHECK(check_match(mat, expected));
}

// Test nested initializer list matrix with single element
TEST_CASE("ParrotTest - NestedMatrixSingleElementTest") {
    // Create a 1x1 matrix
    auto mat = parrot::matrix({{42}});

    // Check size and shape
    CHECK_EQ(mat.size(), 1);
    auto shape = mat.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 1);  // rows
    CHECK_EQ(shape[1], 1);  // cols

    // Check content
    auto expected = parrot::array({42});
    CHECK(check_match(mat, expected));
}

// Test nested initializer list matrix with invalid inputs
TEST_CASE("ParrotTest - NestedMatrixInvalidTest") {
    // Empty nested list - need explicit type since compiler can't deduce from
    // empty list
    std::initializer_list<std::initializer_list<int>> empty_nested{};
    CHECK_THROWS_AS(parrot::matrix(empty_nested), std::invalid_argument);

    // Empty inner list - need explicit type since compiler can't deduce from
    // empty inner list
    std::initializer_list<std::initializer_list<int>> empty_inner{{}};
    CHECK_THROWS_AS(parrot::matrix(empty_inner), std::invalid_argument);

    // Mismatched row lengths
    CHECK_THROWS_AS(parrot::matrix({{1, 2, 3}, {4, 5}}), std::invalid_argument);

    // Another mismatched case
    CHECK_THROWS_AS(parrot::matrix({{1, 2}, {3, 4, 5}, {6, 7}}),
                    std::invalid_argument);
}

// Test transpose function for a 2x3 matrix
TEST_CASE("ParrotTest - Transpose2x3Test") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6})
                 .reshape({2, 3});  // [[1,2,3],[4,5,6]]
    auto transposed = arr.transpose();
    auto expected   = parrot::array({1, 4, 2, 5, 3, 6})
                      .reshape({3, 2});  // [[1,4],[2,5],[3,6]]

    // Check shape
    auto shape = transposed.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 3);
    CHECK_EQ(shape[1], 2);
    CHECK_EQ(transposed.size(), 6);

    // Check content
    CHECK(check_match(transposed, expected));
}

// Test transpose function for a 3x2 matrix (double transpose)
TEST_CASE("ParrotTest - Transpose3x2Test") {
    auto arr = parrot::array({1, 4, 2, 5, 3, 6})
                 .reshape({3, 2});            // [[1,4],[2,5],[3,6]]
    auto transposed_once  = arr.transpose();  // Should be [[1,2,3],[4,5,6]]
    auto transposed_twice = transposed_once
                              .transpose();  // Should be back to original

    // Check shape of single transpose
    auto shape_once = transposed_once.shape();
    REQUIRE_EQ(shape_once.size(), 2);
    CHECK_EQ(shape_once[0], 2);
    CHECK_EQ(shape_once[1], 3);
    CHECK_EQ(transposed_once.size(), 6);

    // Check shape of double transpose
    auto shape_twice = transposed_twice.shape();
    REQUIRE_EQ(shape_twice.size(), 2);
    CHECK_EQ(shape_twice[0], 3);
    CHECK_EQ(shape_twice[1], 2);
    CHECK_EQ(transposed_twice.size(), 6);

    // Check content of double transpose (should match original)
    CHECK(check_match(transposed_twice, arr));
}

// Test transpose function for a single row matrix (1xN)
TEST_CASE("ParrotTest - TransposeSingleRowTest") {
    auto arr = parrot::array({1, 2, 3, 4}).reshape({1, 4});  // [[1,2,3,4]]
    auto transposed = arr.transpose();
    auto expected   = parrot::array({1, 2, 3, 4})
                      .reshape({4, 1});  // [[1],[2],[3],[4]]

    // Check shape
    auto shape = transposed.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 4);
    CHECK_EQ(shape[1], 1);
    CHECK_EQ(transposed.size(), 4);

    // Check content
    CHECK(check_match(transposed, expected));
}

// Test transpose function for a single column matrix (Nx1)
TEST_CASE("ParrotTest - TransposeSingleColumnTest") {
    auto arr = parrot::array({1, 2, 3, 4})
                 .reshape({4, 1});  // [[1],[2],[3],[4]]
    auto transposed = arr.transpose();
    auto expected = parrot::array({1, 2, 3, 4}).reshape({1, 4});  // [[1,2,3,4]]

    // Check shape
    auto shape = transposed.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 1);
    CHECK_EQ(shape[1], 4);
    CHECK_EQ(transposed.size(), 4);

    // Check content
    CHECK(check_match(transposed, expected));
}

// Test transpose function for a 1x1 matrix
TEST_CASE("ParrotTest - Transpose1x1Test") {
    auto arr        = parrot::array({42}).reshape({1, 1});  // [[42]]
    auto transposed = arr.transpose();
    auto expected   = parrot::array({42}).reshape({1, 1});  // [[42]]

    // Check shape
    auto shape = transposed.shape();
    REQUIRE_EQ(shape.size(), 2);
    CHECK_EQ(shape[0], 1);
    CHECK_EQ(shape[1], 1);
    CHECK_EQ(transposed.size(), 1);

    // Check content
    CHECK(check_match(transposed, expected));
}

// Test transpose function with invalid input (1D array)
TEST_CASE("ParrotTest - TransposeInvalid1DTest") {
    auto arr = parrot::array({1, 2, 3, 4});  // 1D array
    CHECK_THROWS_AS(static_cast<void>(arr.transpose()), std::invalid_argument);
}

// Test transpose function with invalid input (3D array)
TEST_CASE("ParrotTest - TransposeInvalid3DTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6, 7, 8})
                 .reshape({2, 2, 2});  // 3D array
    CHECK_THROWS_AS(static_cast<void>(arr.transpose()), std::invalid_argument);
}

// Test scan with Axis=1 (column-wise)
TEST_CASE("ParrotTest - ScanColTest") {
    auto matrix = parrot::array({1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape({3, 3});

    // Column-wise sums
    auto scan_col_sums     = matrix.scan<1>(parrot::add{});
    auto expected_col_sums = parrot::array({1,
                                            2,
                                            3,
                                            1 + 4,
                                            2 + 5,
                                            3 + 6,
                                            1 + 4 + 7,
                                            2 + 5 + 8,
                                            3 + 6 + 9})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_sums, expected_col_sums));

    // Column-wise prods
    auto scan_col_prods     = matrix.scan<1>(parrot::mul{});
    auto expected_col_prods = parrot::array({1,
                                             2,
                                             3,
                                             1 * 4,
                                             2 * 5,
                                             3 * 6,
                                             1 * 4 * 7,
                                             2 * 5 * 8,
                                             3 * 6 * 9})
                                .reshape({3, 3});
    CHECK(check_match(scan_col_prods, expected_col_prods));

    // Column-wise mins
    auto scan_col_mins     = matrix.scan<1>(parrot::min{});
    auto expected_col_mins = parrot::array({1, 2, 3, 1, 2, 3, 1, 2, 3})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_mins, expected_col_mins));

    // Column-wise maxs
    auto scan_col_maxs     = matrix.scan<1>(parrot::max{});
    auto expected_col_maxs = parrot::array({1, 2, 3, 4, 5, 6, 7, 8, 9})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_maxs, expected_col_maxs));
}