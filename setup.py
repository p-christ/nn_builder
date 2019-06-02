import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_builder",
    version="1.0.0",
    author="Petros Christodoulou",
    author_email="p.christodoulou2@gmail.com",
    description="Build neural networks in 1 line",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/p-christ/nn_builder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["tensorflow==2.0.0a0" if sys.platform.startswith("mac") else "tensorflow-gpu==2.0.0a0"]
)