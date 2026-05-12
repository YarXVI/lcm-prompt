"""
LCM v2 PyPI 包配置
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lcm-protocol",
    version="2.0.0",
    author="I.R.I.S. Team",
    author_email="iris@example.com",
    description="Lazy Context Materialization Protocol - 惰性上下文物化协议",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iris-team/lcm-protocol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "mypy>=1.0",
        ],
        "performance": [
            "hnswlib>=0.7",
            "zstandard>=0.21",
            "tiktoken>=0.5",
        ],
        "sqlite": [
            "sqlite3",  # 标准库
        ],
        "multimodal": [
            "PyPDF2>=3.0",
            "Pillow>=10.0",
        ],
        "all": [
            "hnswlib>=0.7",
            "zstandard>=0.21",
            "tiktoken>=0.5",
            "PyPDF2>=3.0",
            "Pillow>=10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lcm=lcm_v2.cli:main",
        ],
    },
)
