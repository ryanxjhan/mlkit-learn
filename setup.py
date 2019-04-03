import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlkit-learn",
    version="0.0.1",
    author="Ryan Han",
    author_email="ryan.han@uwaterloo.ca",
    description="mlkit-learn: lightweight machine learning algorithms for learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryanxjhan/mlkit-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)