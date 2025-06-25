from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="pyshred",
    version="v1.0.21",
    author="David Ye, Jan Williams, Mars Gao, Matteo Tomasetto, Stefano Riva, Nathan Kutz",
    author_email="pyshred1@gmail.com",
    description="PySHRED: Package for Shallow Recurrent Decoding",
    long_description=long_description,
    long_description_content_type="text/markdown",   # 'text/x-rst' for reStructuredText
    packages=find_packages(),
    license = "MIT",
    install_requires=[
        "pysindy",
        "numpy<2.0",
        "scikit-learn",
        "pandas",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
