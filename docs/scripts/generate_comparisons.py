#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import re
import requests
import sys
import tempfile
import subprocess
from pathlib import Path

# Configuration
THRUST_EXAMPLES_URL = "https://github.com/NVIDIA/cccl/tree/main/thrust/examples"
THRUST_RAW_URL = "https://raw.githubusercontent.com/NVIDIA/cccl/main/thrust/examples"
THRUST_API_URL = "https://api.github.com/repos/NVIDIA/cccl/contents/thrust/examples"
PARROT_THRUST_DIR = "examples/thrust"
THRUST_OUTPUT_FILE = "docs/parrot_v_thrust.rst"

GETTING_STARTED_EXAMPLES_DIR = "examples/getting_started"
REAL_WORLD_EXAMPLES_DIR = "examples/real_world"
EXAMPLES_OUTPUT_FILE = "docs/examples.rst"

# Real world examples are configured via comments in the Parrot files themselves
# Expected format:
# // https://github.com/{owner}/{repo}/blob/{commit}/{path}#L{start_line}
# // to {end_line}


def get_examples_list(api_url):
    """Fetch list of example files from a GitHub repository."""
    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error fetching examples: {response.status_code}")
            return []

        examples = []
        for item in response.json():
            if (item["type"] == "file" and item["name"].endswith(".cu")
                    and not item["name"].startswith("_")):
                examples.append(item["name"][:-3])  # Remove .cu extension
        return examples
    except Exception as e:
        print(f"Error fetching examples: {e}")
        return []


def format_code_with_clang(code):
    """Format code using clang-format and the project's .clang-format file."""
    if not code:
        return None

    # Check if clang-format is available
    try:
        subprocess.run(["clang-format", "--version"],
                       capture_output=True,
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # clang-format not available, return original code
        return code

    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w+") as tmp_file:
            # Write the unformatted code to the temporary file
            tmp_file.write(code)
            tmp_file.flush()

            # Run clang-format on the temporary file
            result = subprocess.run(
                ["clang-format", tmp_file.name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Return the formatted code
            return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running clang-format: {e}")
        return code  # Return original code if formatting fails
    except Exception as e:
        print(f"Error formatting code: {e}")
        return code


def fetch_example_content(raw_url, example_name):
    """Fetch the content of an example from GitHub."""
    url = f"{raw_url}/{example_name}.cu"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        print(f"Failed to fetch {url}, status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching example: {e}")
        return None


def fetch_file_from_github(owner, repo, branch, file_path):
    """Fetch a file from a GitHub repository."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    try:
        print(f"Fetching from: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        print(f"Failed to fetch {url}, status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching file from GitHub: {e}")
        return None


def parse_github_url_from_comments(code):
    """
    Parse GitHub URL and line range from comments in the code.
    Expected format:
    // https://github.com/{owner}/{repo}/blob/{commit}/{path}#L{start_line}
    // to {end_line}
    
    Returns a dict with: url, owner, repo, commit, path, start_line, end_line
    """
    lines = code.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('//') and 'github.com' in line:
            # Extract URL from comment
            url = line[2:].strip()

            # Check if next line has "to {end_line}"
            end_line = None
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('//') and 'to' in next_line:
                    # Extract end line number
                    parts = next_line[2:].strip().split()
                    if len(parts) >= 2 and parts[0].lower() == 'to':
                        try:
                            end_line = int(parts[1])
                        except ValueError:
                            pass

            # Parse the GitHub URL
            # Format: https://github.com/{owner}/{repo}/blob/{commit}/{path}#L{start_line}
            if '#L' in url:
                base_url, line_part = url.split('#L')
                try:
                    start_line = int(line_part)
                except ValueError:
                    continue

                # Parse the base URL
                # Remove https://github.com/
                if 'github.com/' in base_url:
                    path_parts = base_url.split('github.com/')[1]
                    parts = path_parts.split('/')

                    if len(parts) >= 4 and parts[2] == 'blob':
                        owner = parts[0]
                        repo = parts[1]
                        commit = parts[3]
                        file_path = '/'.join(parts[4:])

                        return {
                            'url': url,
                            'owner': owner,
                            'repo': repo,
                            'commit': commit,
                            'path': file_path,
                            'start_line': start_line,
                            'end_line': end_line
                        }

    return None


def extract_lines_from_code(code, start_line, end_line=None):
    """Extract specific lines from code (1-indexed)."""
    if not code:
        return None

    lines = code.split('\n')

    # Convert to 0-indexed
    start_idx = start_line - 1

    if end_line is None:
        # Just get the single line
        if 0 <= start_idx < len(lines):
            return lines[start_idx]
        return None

    # Get range of lines
    end_idx = end_line
    if 0 <= start_idx < len(lines):
        return '\n'.join(lines[start_idx:end_idx])

    return None


def extract_function_from_code(code, function_name):
    """
    Extract a specific function from code.
    This is a simple extraction that looks for template/function definitions.
    """
    if not code or not function_name:
        return code

    lines = code.split('\n')
    result_lines = []
    in_function = False
    brace_count = 0
    function_started = False
    paren_count = 0
    found_opening_brace = False

    for i, line in enumerate(lines):
        # Look for function name in the line
        if function_name in line and not in_function:
            # Check if this is a function declaration (has opening parenthesis)
            if '(' in line:
                # Look backwards for comments, templates, or return types
                start_idx = i
                for j in range(i - 1, max(-1, i - 20), -1):
                    stripped = lines[j].strip()
                    # Include comment lines, template lines, and return type lines
                    if (stripped.startswith('//') or stripped.startswith('/*')
                            or 'template' in stripped
                            or (stripped and not stripped.endswith(';')
                                and not stripped.endswith('}'))):
                        start_idx = j
                    else:
                        break

                # Add lines from start
                for j in range(start_idx, i + 1):
                    result_lines.append(lines[j])

                in_function = True
                function_started = True

                # Count parentheses and braces
                for ch in line:
                    if ch == '(':
                        paren_count += 1
                    elif ch == ')':
                        paren_count -= 1
                    elif ch == '{':
                        brace_count += 1
                        found_opening_brace = True
                    elif ch == '}':
                        brace_count -= 1
                continue

        if in_function:
            result_lines.append(line)

            # Count parentheses and braces
            for ch in line:
                if ch == '(':
                    paren_count += 1
                elif ch == ')':
                    paren_count -= 1
                elif ch == '{':
                    brace_count += 1
                    found_opening_brace = True
                elif ch == '}':
                    brace_count -= 1

            # Check if we've reached the end
            # For function declarations (no braces), stop at semicolon
            if not found_opening_brace and line.strip().endswith(
                    ';') and paren_count == 0:
                break

            # For function definitions (with braces), stop when braces are balanced
            if found_opening_brace and brace_count == 0:
                break

    if result_lines:
        return '\n'.join(result_lines)

    # If extraction failed, return original code
    return code


def read_parrot_example(filename):
    """Read a Parrot example file."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading Parrot example {filename}: {e}")
        return None


def strip_license_header(code):
    """
    Strip SPDX license header and GitHub URL comments from code.
    Removes the block comment at the start of the file and any GitHub URL comments.
    """
    if not code:
        return code

    lines = code.split('\n')
    result_lines = []
    in_license_block = False
    found_first_code = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check if we're entering a license block comment
        if not found_first_code and stripped.startswith('/*'):
            in_license_block = True
            continue

        # Check if we're exiting a license block comment
        if in_license_block and '*/' in line:
            in_license_block = False
            continue

        # Skip lines inside license block
        if in_license_block:
            continue

        # Skip GitHub URL comments (after license block)
        if not found_first_code and stripped.startswith('//'):
            # Check if it's a GitHub URL or "to" line
            if 'github.com' in stripped or stripped.startswith('// to'):
                continue

        # Skip empty lines at the beginning
        if not found_first_code and not stripped:
            continue

        # We've found the first line of actual code
        found_first_code = True
        result_lines.append(line)

    return '\n'.join(result_lines)


def strip_comments_and_count_lines(code):
    """Strip comments from code and count non-empty lines."""
    if not code:
        return 0

    # Remove C-style comments (both /* */ and // style)
    code_no_block_comments = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code_no_comments = re.sub(r"//.*$",
                              "",
                              code_no_block_comments,
                              flags=re.MULTILINE)

    # Count non-empty lines
    non_empty_lines = [
        line.strip() for line in code_no_comments.splitlines() if line.strip()
    ]
    return len(non_empty_lines)


def write_rst_file(examples, output_file, comparison_lib):
    """Write the RST file with the examples."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"Parrot vs {comparison_lib}\n")
            f.write("=" * (len(f"Parrot vs {comparison_lib}") + 1) + "\n\n")
            f.write(
                f"This page provides comparisons between Parrot code and equivalent {comparison_lib} code for common operations. "
            )

            if comparison_lib == "Thrust":
                f.write(
                    "Parrot is built on top of Thrust but provides a more concise and expressive API.\n\n"
                )
            else:
                f.write(
                    f"This comparison demonstrates how Parrot's API compares to {comparison_lib}'s API.\n\n"
                )

            f.write(".. contents:: Examples\n")
            f.write("   :local:\n")
            f.write("   :depth: 1\n\n")

            f.write("Examples\n")
            f.write("--------\n\n")
            f.write("|\n")

            for example in examples:
                if example["parrot_code"] and example["comparison_code"]:
                    # Create title with completion status
                    status = "✅" if example.get(
                        "status") == "complete" else "🟡"
                    title_text = f"{status} {example['title']}"

                    # Add code ratio if available
                    if (example.get("code_ratio")
                            and example.get("status") == "complete"):
                        title_text += f" ({example['code_ratio']}x less code)"

                    f.write(f"{title_text}\n")
                    f.write("~" * (len(title_text) + 1) + "\n\n")

                    # Write Parrot Code section
                    f.write("**Parrot Code**\n\n")
                    f.write(".. code-block:: cpp\n\n")
                    for line in example["parrot_code"].splitlines():
                        f.write(f"    {line}\n")
                    f.write("\n")

                    # Write Comparison Code section
                    f.write(f"**{comparison_lib} Code**\n\n")
                    if example.get("status") == "complete":
                        if comparison_lib == "Thrust":
                            f.write(
                                f"Link: https://github.com/NVIDIA/cccl/blob/main/thrust/examples/{example['comparison_filename']}.cu\n\n"
                            )

                        f.write(".. code-block:: cpp\n\n")
                        for line in example["comparison_code"].splitlines():
                            f.write(f"    {line}\n")
                    else:
                        f.write(".. code-block:: cpp\n\n")
                        f.write("    // TODO\n")
                    f.write("\n")
        return True
    except Exception as e:
        print(f"Error writing RST file: {e}")
        return False


def auto_generate_title(filename):
    """Generate a title from a filename."""
    if filename == "minmax.cu":
        return "Min Max"
    words = filename.replace(".cu", "").replace("_", " ").split()
    return " ".join(word.capitalize() for word in words)


def check_example_status(parrot_code, comparison_code):
    """Determine if the example is complete or needs work."""
    if "TODO" in parrot_code or "TODO" in comparison_code:
        return "incomplete"
    return "complete"


def process_examples(parrot_dir, raw_url, api_url, github_repo_url, lib_name):
    """Process examples for a given library comparison."""
    print(f"Processing {lib_name} examples...")

    # Check if examples directory exists
    if not os.path.isdir(parrot_dir):
        print(f"Error: Directory '{parrot_dir}' not found!")
        return []

    # Get list of examples
    remote_examples = get_examples_list(api_url)
    if not remote_examples:
        print(
            f"Warning: Could not fetch {lib_name} examples list. Using filenames only..."
        )

    examples = []

    # Discover Parrot examples
    parrot_files = [f for f in os.listdir(parrot_dir) if f.endswith(".cu")]

    for parrot_file in parrot_files:
        print(f"Processing {parrot_file}...")
        parrot_path = os.path.join(parrot_dir, parrot_file)
        parrot_basename = os.path.splitext(parrot_file)[0]

        # For simplicity, use the same basename for the comparison example
        comparison_filename = parrot_basename

        # Generate title from filename
        title = auto_generate_title(parrot_file)

        # Default to complete unless we find out otherwise
        status = "complete"

        example = {
            "title": title,
            "comparison_filename": comparison_filename,
            "status": status,
            "parrot_code": None,
            "comparison_code": None,
        }

        # Read Parrot example
        parrot_code = read_parrot_example(parrot_path)
        if not parrot_code:
            print(f"Warning: Could not read Parrot example '{parrot_path}'!")
            continue

        # Strip license header for display
        example["parrot_code"] = strip_license_header(parrot_code)

        # Fetch comparison example
        comparison_code = fetch_example_content(raw_url, comparison_filename)
        if not comparison_code:
            print(
                f"Warning: Could not fetch {lib_name} example '{comparison_filename}'!"
            )
            example["status"] = "incomplete"
            example["comparison_code"] = "// TODO"
        else:
            # Format comparison code with clang-format
            example["comparison_code"] = format_code_with_clang(
                comparison_code)

        # Update status based on content
        if example["parrot_code"] and example["comparison_code"]:
            example["status"] = check_example_status(
                example["parrot_code"], example["comparison_code"])

            # Calculate code ratio if both examples are complete
            if example["status"] == "complete":
                parrot_lines = strip_comments_and_count_lines(
                    example["parrot_code"])
                comparison_lines = strip_comments_and_count_lines(
                    example["comparison_code"])

                if parrot_lines > 0 and comparison_lines > 0:
                    ratio = comparison_lines / parrot_lines
                    example["code_ratio"] = round(ratio, 1)
                    print(
                        f"  Code ratio for {example['title']}: {example['code_ratio']}x"
                    )

        examples.append(example)

    # Sort examples alphabetically by title
    examples.sort(key=lambda x: x["title"])

    return examples


def process_getting_started_examples(getting_started_dir):
    """Process standalone Parrot examples from getting_started directory."""
    print("Processing getting_started examples...")

    # Check if examples directory exists
    if not os.path.isdir(getting_started_dir):
        print(f"Error: Directory '{getting_started_dir}' not found!")
        return []

    examples = []

    # Discover getting_started examples
    getting_started_files = [
        f for f in os.listdir(getting_started_dir) if f.endswith(".cu")
    ]

    for getting_started_file in getting_started_files:
        print(f"Processing {getting_started_file}...")
        getting_started_path = os.path.join(getting_started_dir,
                                            getting_started_file)
        getting_started_basename = os.path.splitext(getting_started_file)[0]

        # Generate title from filename
        title = auto_generate_title(getting_started_file)

        example = {
            "title": title,
            "filename": getting_started_basename,
            "code": None,
        }

        # Read getting_started example
        code = read_parrot_example(getting_started_path)
        if not code:
            print(
                f"Warning: Could not read getting_started example '{getting_started_path}'!"
            )
            continue

        # Strip license header for display
        example["code"] = strip_license_header(code)

        examples.append(example)

    # Sort examples alphabetically by title
    examples.sort(key=lambda x: x["title"])

    return examples


def process_real_world_examples(real_world_dir):
    """Process real world examples from real_world directory."""
    print("Processing real world examples...")

    # Check if examples directory exists
    if not os.path.isdir(real_world_dir):
        print(f"Error: Directory '{real_world_dir}' not found!")
        return []

    examples = []

    # Discover subdirectories in real_world
    subdirs = [
        d for d in os.listdir(real_world_dir)
        if os.path.isdir(os.path.join(real_world_dir, d))
    ]

    for subdir in subdirs:
        print(f"Processing real world example: {subdir}...")
        subdir_path = os.path.join(real_world_dir, subdir)

        # Find all .cu and .h files in the subdirectory (Parrot versions)
        parrot_files = [
            f for f in os.listdir(subdir_path)
            if f.endswith('.cu') or f.endswith('.h')
        ]

        for parrot_file in parrot_files:
            parrot_path = os.path.join(subdir_path, parrot_file)

            # Read Parrot example
            parrot_code = read_parrot_example(parrot_path)
            if not parrot_code:
                print(f"Warning: Could not read parrot file '{parrot_path}'!")
                continue

            # Parse GitHub URL from comments
            github_info = parse_github_url_from_comments(parrot_code)
            if not github_info:
                print(
                    f"Warning: No GitHub URL found in '{parrot_path}'! Skipping."
                )
                continue

            print(f"  Found GitHub reference: {github_info['url']}")

            # Fetch original code from GitHub using commit hash
            original_code = fetch_file_from_github(github_info['owner'],
                                                   github_info['repo'],
                                                   github_info['commit'],
                                                   github_info['path'])

            if not original_code:
                print(f"Warning: Could not fetch original code from GitHub!")
                continue

            # Extract specific line range if specified
            if github_info['start_line'] and github_info['end_line']:
                print(
                    f"  Extracting lines {github_info['start_line']} to {github_info['end_line']}"
                )
                original_code = extract_lines_from_code(
                    original_code, github_info['start_line'],
                    github_info['end_line'])

            if not original_code:
                print(f"Warning: Could not extract lines from original code!")
                continue

            # Format the original code with clang-format
            original_code = format_code_with_clang(original_code)

            # Generate title from subdirectory name
            title = subdir.replace('_', ' ').title()

            # Build the full GitHub URL for the source (with line numbers if available)
            source_url = github_info['url']

            # Strip license header from Parrot code for display
            parrot_code_display = strip_license_header(parrot_code)

            # Create the example entry
            example = {
                "title": title,
                "subdir": subdir,
                "original_code": original_code,
                "parrot_code": parrot_code_display,
                "source_url": source_url,
            }

            # Calculate code ratio
            original_lines = strip_comments_and_count_lines(original_code)
            parrot_lines = strip_comments_and_count_lines(parrot_code)

            if original_lines > 0 and parrot_lines > 0:
                ratio = original_lines / parrot_lines
                example["code_ratio"] = round(ratio, 1)
                print(
                    f"  Code ratio for {example['title']}: {example['code_ratio']}x reduction"
                )

            examples.append(example)

    # Sort examples alphabetically by title
    examples.sort(key=lambda x: x["title"])

    return examples


def write_examples_rst_file(examples, real_world_examples, output_file):
    """Write the examples RST file with standalone Parrot examples and real world examples."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("Examples\n")
            f.write("========\n\n")
            f.write(
                "This page provides standalone Parrot examples demonstrating common operations and patterns, "
                "as well as real world examples showing before and after code transformations.\n\n"
            )

            f.write(".. contents:: Examples\n")
            f.write("   :local:\n")
            f.write("   :depth: 1\n\n")

            # Write standalone examples section
            if examples:
                f.write("Getting Started Examples\n")
                f.write("------------------------\n\n")
                f.write(
                    "These examples demonstrate basic Parrot functionality and common patterns.\n\n"
                )

                for example in examples:
                    if example["code"]:
                        title_text = example["title"]

                        f.write(f"{title_text}\n")
                        f.write("~" * len(title_text) + "\n\n")

                        f.write(".. code-block:: cpp\n\n")
                        for line in example["code"].splitlines():
                            f.write(f"    {line}\n")
                        f.write("\n")

            # Write real world examples section
            if real_world_examples:
                f.write("Real World Examples\n")
                f.write("-------------------\n\n")
                f.write(
                    "These examples show real world code transformations using Parrot. "
                    "Each example demonstrates the before and after code, highlighting "
                    "how Parrot simplifies complex operations.\n\n")

                for example in real_world_examples:
                    if example["original_code"] and example["parrot_code"]:
                        # Create title with code reduction ratio
                        title_text = example["title"]
                        if example.get("code_ratio"):
                            title_text += f" ({example['code_ratio']}x code reduction)"

                        f.write(f"{title_text}\n")
                        f.write("~" * len(title_text) + "\n\n")

                        # Add source URL if available
                        if example.get("source_url"):
                            f.write(
                                f"**Source:** `Original Code <{example['source_url']}>`_\n\n"
                            )

                        # Write Before (Original) Code section
                        f.write("**Before (Original)**\n\n")
                        f.write(".. code-block:: cpp\n\n")
                        for line in example["original_code"].splitlines():
                            f.write(f"    {line}\n")
                        f.write("\n")

                        # Write After (Parrot) Code section
                        f.write("**After (Parrot)**\n\n")
                        f.write(".. code-block:: cpp\n\n")
                        for line in example["parrot_code"].splitlines():
                            f.write(f"    {line}\n")
                        f.write("\n")
        return True
    except Exception as e:
        print(f"Error writing examples RST file: {e}")
        return False


def main():
    # Process Thrust examples
    thrust_examples = process_examples(PARROT_THRUST_DIR, THRUST_RAW_URL,
                                       THRUST_API_URL, THRUST_EXAMPLES_URL,
                                       "Thrust")

    # Write the Thrust RST file
    if thrust_examples:
        if write_rst_file(thrust_examples, THRUST_OUTPUT_FILE, "Thrust"):
            print(
                f"Documentation generated successfully: {THRUST_OUTPUT_FILE}")
        else:
            print(f"Thrust documentation generation failed!")
    else:
        print("No Thrust examples were processed!")

    # Process getting_started examples
    getting_started_examples = process_getting_started_examples(
        GETTING_STARTED_EXAMPLES_DIR)

    # Process real world examples
    real_world_examples = process_real_world_examples(REAL_WORLD_EXAMPLES_DIR)

    # Write the examples RST file
    if getting_started_examples or real_world_examples:
        if write_examples_rst_file(getting_started_examples,
                                   real_world_examples, EXAMPLES_OUTPUT_FILE):
            print(
                f"Examples documentation generated successfully: {EXAMPLES_OUTPUT_FILE}"
            )
        else:
            print(f"Examples documentation generation failed!")
    else:
        print("No examples were processed!")


if __name__ == "__main__":
    main()
