#!/usr/bin/env python

"""
Minimal script to download NLTK data
"""

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Only import the downloader module
from nltk.downloader import download

# Download to a local directory
download_dir = './nltk_data'

# Download each resource
print("Downloading wordnet...")
download('wordnet', download_dir=download_dir)

print("Downloading punkt...")
download('punkt', download_dir=download_dir)

print("Downloading stopwords...")
download('stopwords', download_dir=download_dir)

print("Downloading omw-1.4...")
download('omw-1.4', download_dir=download_dir)

print("All NLTK resources downloaded successfully.")
