from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qtsuru",
    version="0.1.0",
    author="Lucas Rattighieri",
    author_email="lucas.rattighieri@gmail.com",
    description="Quantum circuit and algorithm simulation library using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lucas-Rattighieri/qsim",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
