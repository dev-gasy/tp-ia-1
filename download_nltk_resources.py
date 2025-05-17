#!/usr/bin/env python

"""
A simple script to download NLTK resources without dependencies
"""

import nltk

# Set NLTK data path explicitly
nltk.data.path.append('./nltk_data')

# Download resources one by one
print("Downloading wordnet...")
nltk.download('wordnet', download_dir='./nltk_data')

print("Downloading punkt...")
nltk.download('punkt', download_dir='./nltk_data')

print("Downloading stopwords...")
nltk.download('stopwords', download_dir='./nltk_data')

print("Downloading omw-1.4...")
nltk.download('omw-1.4', download_dir='./nltk_data')

print("All NLTK resources downloaded successfully.")
