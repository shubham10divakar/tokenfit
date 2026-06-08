"""Packaging for tokenfit.

Build:   python -m build         (produces wheel + sdist in dist/)
Install: pip install .           (or `pip install -e .` for development)
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="tokenfit",
    version="1.1.0",
    description="Fit your whole repo into any small model's token window — "
    "context selection for free/small LLMs.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Shubham Divakar",
    author_email="shubham.divakar@gmail.com",
    url="https://github.com/shubham10divakar/tokenfit",
    project_urls={
        "Source": "https://github.com/shubham10divakar/tokenfit",
        "Issues": "https://github.com/shubham10divakar/tokenfit/issues",
        "Examples": "https://github.com/shubham10divakar/tokenfit/blob/main/EXAMPLES.md",
    },
    license="MIT",
    packages=find_packages(include=["tokenfit", "tokenfit.*"]),
    include_package_data=True,
    package_data={"tokenfit": ["eval/dataset/*.yaml"]},
    python_requires=">=3.9",
    install_requires=[
        "huggingface_hub>=0.25.0",
        "transformers>=4.44.0",
        "sentence-transformers>=3.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "rank-bm25>=0.2.2",  # Phase 2: hybrid (semantic + keyword) retrieval, default on
    ],
    extras_require={
        # Phase 3 swap: scalable vector store
        "chroma": ["chromadb>=0.5.0"],
        "dev": ["pytest>=8.0", "build>=1.2"],
    },
    entry_points={
        "console_scripts": [
            "tokenfit=tokenfit.cli:main",
            "tokenfit-eval=tokenfit.eval.harness:main",
        ],
    },
    keywords=["llm", "rag", "context", "huggingface", "coding-agent", "retrieval"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
