#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download NLTK Wordnet Resource
"""

import ssl

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK resource: wordnet")
nltk.download('wordnet')
print("Download complete!")
