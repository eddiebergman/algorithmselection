import os
import setuptools
import sys

if sys.version_info < (3, 8):
    raise ValueError(f"Unsupported Python version {sys.version_info.major}."
                     + f"{sys.version_info.micro}.{sys.version_info.micro}"
                     + f"found. Requires Python 3.8 or higher."
    )

with open(os.path.join("README.md")) as fid:
    README = fid.read()

setuptools.setup(
    name="algorithmselection",
    author="Eddie Bergman",
    author_email="Edward.Bergman@uni-siegen.de",
    maintainer="Eddie Bergman",
    maintainer_email="Edward.Bergman@uni-siegen.de",
    description="A module for Ensemble Based Algorithm Selection",
    long_description=README,
    long_description_content_type="text/markdown",
    license="BSD 3-clause",
    version="0.1.0",
    packages=setuptools.find_packages(
        include=["algorithmselection.*", "algorithmselection"], 
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"": ["*.md", "py.typed"]},
    python_requires=">=3.8",
    install_requires=[
        "auto-sklearn==0.10.0",
        "numpy>=1.19.2",
        "pandas<1.0",
        "scikit-learn>=0.22.0,<0.23", 
        "openml>=0.10.2",
        "scikit-optimize",
        "pyDOE",
        "emcee",
    ],
    extras_require={
        "test": [
            "pytest",
            "mypy",
            "pylint",
            "ipython"
        ],
    },
    test_suite="pytest",
)
