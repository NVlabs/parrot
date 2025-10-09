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

# Configuration file for the Sphinx documentation builder
# -- Project information -----------------------------------------------------
project = "Parrot"
# copyright = "2025, Conor Hoekstra"
author = "Conor Hoekstra"

# Logo configuration
html_logo = "_static/logo.png"

# Furo theme options
html_theme_options = {
    "sidebar_hide_name":
    False,  # Set to True if you want to hide the project name next to logo
    "light_css_variables": {
        # Visual Studio Light theme colors
        "color-foreground-primary": "#1e1e1e",
        "color-foreground-secondary": "#5c5c5c",
        "color-foreground-muted": "#6e6e6e",
        "color-foreground-tertiary": "#757575",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f3f3f3",
        "color-background-hover": "#e8e8e8",
        "color-brand-primary": "#0066b8",
        "color-brand-content": "#0066b8",
        "color-brand-secondary": "#007acc",
        "color-brand-tertiary": "#42a5f5",
        "color-brand-quaternary": "#66bb6a",
        "color-brand-quinary": "#ffb74d",
        "color-brand-senary": "#ec407a",
        "color-brand-septenary": "#4db6ac",
        "color-brand-octonary": "#ba68c8",
        "color-api-background": "#f3f3f3",
        "color-api-foreground": "#333333",
        "color-sidebar-background": "#f3f3f3",
        "color-sidebar-item-background--current": "#e6f0fa",
        "color-sidebar-item-background--hover": "#e3e3e3",
        "color-sidebar-item-expander-background--hover": "#e3e3e3",
        "color-link": "#0066b8",
        "color-link--hover": "#005cb8",
        "color-inline-code-background": "#f0f0f0",
    },
    "dark_css_variables": {
        # Dracula Soft theme colors
        "color-foreground-primary": "#f8f8f2",
        "color-foreground-secondary": "#bdae93",
        "color-foreground-muted": "#a89984",
        "color-foreground-tertiary": "#a89984",
        "color-background-primary": "#282a36",
        "color-background-secondary": "#383a59",
        "color-background-hover": "#44475a",
        "color-brand-primary": "#bd93f9",
        "color-brand-content": "#bd93f9",
        "color-brand-secondary": "#ff79c6",
        "color-brand-tertiary": "#8be9fd",
        "color-brand-quaternary": "#50fa7b",
        "color-brand-quinary": "#ffb86c",
        "color-brand-senary": "#ff5555",
        "color-brand-septenary": "#6272a4",
        "color-brand-octonary": "#f1fa8c",
        "color-api-background": "#383a59",
        "color-api-foreground": "#f8f8f2",
        "color-sidebar-background": "#343746",
        "color-sidebar-item-background--current": "#44475a",
        "color-sidebar-item-background--hover": "#3a3c4e",
        "color-sidebar-item-expander-background--hover": "#3a3c4e",
        "color-link": "#ff79c6",
        "color-link--hover": "#ff92df",
        "color-inline-code-background": "#44475a",
    },
    "navigation_with_keys": True,
    "source_repository": "https://github.com/NVlabs/parrot/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# To improve navigation of class methods, add this:
# Tell Sphinx to generate detailed cross-reference targets
add_module_names = False
python_use_unqualified_type_names = True

# -- General configuration ---------------------------------------------------
from exhale import utils

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "breathe",
    "exhale",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"  # Modern, clean theme
html_static_path = ["_static"]

# Configure Pygments for code highlighting
pygments_style = "default"  # Base style, will be overridden by custom CSS
pygments_dark_style = "dracula"  # Dark style, will be overridden by custom CSS

# HTML context to pass variables to templates
html_context = {
    "fusion_array_page": "fusion_array",
}

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {"parrot": "./doxyoutput/xml/"}
breathe_default_project = "parrot"
breathe_domain_by_extension = {
    "hpp": "cpp",
    "cu": "cpp",
}

# Make sure all methods appear in the index
breathe_default_members = ("members", "undoc-members")
breathe_show_include = False
breathe_show_enumvalue_initializer = True

# Add custom CSS
html_css_files = [
    "dark_code_fix.css",
    "custom.css",
    "pygments_custom.css",  # Custom Pygments styling
]

# Enable better sidebar structure
html_show_sourcelink = False
html_sidebars = {
    "**": ["sidebar.html"],  # Use our custom sidebar for all pages
    "fusion_array": ["sidebar.html"],  # Especially for the fusion_array page
}

# Exhale configuration
exhale_args = {
    # Required arguments
    "containmentFolder":
    "./api",
    "rootFileName":
    "library_root.rst",
    "doxygenStripFromPath":
    "..",
    "rootFileTitle":
    "Parrot API",
    # Suggested optional arguments
    "createTreeView":
    True,
    "treeViewIsBootstrap":
    False,
    "exhaleExecutesDoxygen":
    True,
    # Generate more detailed class hierarchies in the sidebar
    "fullApiSubSectionTitle":
    "Full API",
    "afterTitleDescription":
    "Complete API documentation for all classes and functions.",
    "fullToctreeMaxDepth":
    5,
    "includeTemplateParamOrderList":
    True,
    "unabridgedOrphanKinds": ["class", "function", "define", "file"],
    "verboseBuild":
    True,
    # Additional options to expose methods in sidebar
    "contentsDirectives":
    True,
    "kindsWithContentsDirectives": ["class", "namespace"],
    # Doxygen settings
    "exhaleDoxygenStdin":
    """
INPUT                  = ../parrot.hpp
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
RECURSIVE              = YES
PREDEFINED             = __host__= __device__=
GENERATE_XML           = YES
XML_PROGRAMLISTING     = YES
JAVADOC_AUTOBRIEF      = YES
BRIEF_MEMBER_DESC      = YES
REPEAT_BRIEF           = YES
EXCLUDE_SYMBOLS        = parrot::add_functor parrot::append_functor parrot::delta parrot::neq parrot::double_functor parrot::min_functor parrot::times_functor
""",
}
