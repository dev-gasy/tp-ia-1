#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup file for the Insurance Claim Duration Prediction project.
This file helps manage dependencies and installation of the project.
"""

from setuptools import setup, find_packages

setup(
    name="insurance-claims-duration",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn==1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk==3.8.1",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.0",
    ],
    extras_require={
        "dev": [
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "pre-commit>=2.13.0",
        ],
        "test": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "pytest-xdist>=2.3.0",
            "httpx>=0.18.2",  # For FastAPI testing
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    description="A system for predicting insurance claim duration",
    author="Tolotra RAHARISON",
    author_email="tolotra.raharison@example.com",
)
