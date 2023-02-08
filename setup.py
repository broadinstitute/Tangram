#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

d = {}
with open("tangram/_version.py") as f:
    exec(f.read(), d)

setuptools.setup(
    name="tangram-sc",
    version=d["__version__"],
    author="Tommaso Biancalani, Gabriele Scalia",
    author_email="tommaso.biancalani@gmail.com",
    description="Spatial alignment of single cell transcriptomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/broadinstitute/Tangram",
    packages=setuptools.find_namespace_packages(),
    classifiers=["Programming Language :: Python :: 3.8", "Operating System :: MacOS",],
    python_requires=">=3.8.5",
    install_requires=[
        "pip",
        "torch",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "scanpy",
        "scikit-learn",
        "tqdm",
    ],
)

