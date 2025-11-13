"""Setup script for Enhanced PrivBayes."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="privbayes-enhanced",
    version="1.0.0",
    author="Narasimha Raghavan Veeraragavan",
    author_email="vnragavan@protonmail.com",
    description="Enhanced PrivBayes: Differentially Private Bayesian Network Synthesizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vnragavan/privbayes-enhanced",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'privbayes=privbayes_enhanced.cli:main',
        ],
    },
)


