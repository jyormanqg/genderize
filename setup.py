from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Get version from __init__.py
def get_version():
    with open("latam_genderize/__init__.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"

setup(
    name="latam-gender-predictor",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for predicting gender based on Latin American names using machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/latam-gender-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.0.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "unidecode>=1.0.0",
        "pyspark>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    package_data={
        "latam_genderize": ["models/*.h5"],
    },
    zip_safe=False,
    keywords="gender prediction, machine learning, latin america, names, tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/latam-gender-predictor/issues",
        "Source": "https://github.com/yourusername/latam-gender-predictor",
        "Documentation": "https://github.com/yourusername/latam-gender-predictor#readme",
    },
) 
