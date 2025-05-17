#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock utilities for testing without requiring actual NLTK resources.
"""


def clean_text_mock(text):
    """
    Mock clean_text function that performs minimal cleaning.
    
    Args:
        text: Text to clean and normalize
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace special characters with spaces
    import re
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def calculate_age_mock(year_disability, year_birth):
    """
    Calculate age at time of disability
    
    Args:
        year_disability: Year when disability started
        year_birth: Year of birth
        
    Returns:
        Age in years
    """
    return year_disability - year_birth
