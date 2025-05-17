#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download required NLTK resources for the project.
"""

import subprocess
import sys


def download_nltk_resources():
    """Download all necessary NLTK resources using subprocess."""
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'  # Open Multilingual WordNet
    ]

    for resource in resources:
        print(f"Downloading NLTK resource: {resource}")
        python_code = f"import nltk; nltk.download('{resource}')"
        subprocess.run([sys.executable, '-c', python_code], check=True)
        print(f"Successfully downloaded {resource}")


if __name__ == "__main__":
    download_nltk_resources()
    print("All NLTK resources downloaded successfully.")
