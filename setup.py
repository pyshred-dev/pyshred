from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="pyshred",
    version="1.0.4",
    author="Kutz Research Group",
    author_email="pyshred1@gmail.com",
    description="PySHRED: A Python Package for SHallow REcurrent Decoders (SHRED) for Spatial-Temporal Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",   # 'text/x-rst' for reStructuredText
    packages=find_packages(),
    license = "BSD-3-Clause",
    install_requires=[
        "numpy>=1.21,<2.0",
        "scipy<1.13",
        "pandas",
        "torch",
        "scikit-learn",
        "pysindy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
