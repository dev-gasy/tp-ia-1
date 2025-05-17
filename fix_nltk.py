#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix NLTK dependency issues by using mocks
"""

# Import the mock setup function before anything else
from tests.mocks.nltk_mock import setup_nltk_mock

# Setup the mock NLTK
print("Setting up NLTK mock...")
setup_nltk_mock()
print("NLTK mock setup complete.")

# Run the main script
if __name__ == "__main__":
    import sys
    import main

    sys.exit(main.main())
