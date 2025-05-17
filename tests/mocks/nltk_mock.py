#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock NLTK module for testing without requiring actual NLTK resources.
"""

import sys
import types


class MockWordNetLemmatizer:
    """Mock WordNet Lemmatizer."""

    def lemmatize(self, word, pos='n'):
        """Mock lemmatize method that returns the input word unchanged."""
        return word


def download_mock(resource_name, quiet=False, download_dir=None):
    """Mock download function that does nothing."""
    if not quiet:
        print(f"Mock downloading NLTK resource: {resource_name}")
    return True


def word_tokenize(text):
    """Mock word tokenizer that simply splits on spaces."""
    return text.split()


def clean_text_mock(text):
    """
    Mock clean text function that performs minimal cleaning.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    return text.lower()


# Add these additional mock components

class MockData:
    """Mock NLTK Data class"""
    path = ['/mock/nltk/data']

    @staticmethod
    def find(resource_name):
        """Mock find method"""
        return f"/mock/nltk/data/{resource_name}"


class MockTranslate:
    """Mock Translate module"""
    meteor = lambda x, y: 0.5  # A placeholder meteor score


# WordNet specific mocks
class MockWordNet:
    """Mock WordNet class"""

    @staticmethod
    def morphy(word, pos=None):
        """Mock morphy function"""
        return word


# Stopwords mock
class MockStopwords:
    @staticmethod
    def words(language):
        return ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to']


# Create a full mock of NLTK and all its expected submodules
def setup_nltk_mock():
    """Sets up a comprehensive mock of NLTK and installs it in sys.modules"""
    # Create the main NLTK mock module
    nltk_mock = types.ModuleType('nltk')
    nltk_mock.download = download_mock
    nltk_mock.word_tokenize = word_tokenize

    # Set up tokenize module
    tokenize_mock = types.ModuleType('nltk.tokenize')
    tokenize_mock.word_tokenize = word_tokenize
    nltk_mock.tokenize = tokenize_mock
    sys.modules['nltk.tokenize'] = tokenize_mock

    # Set up data submodule
    data_mock = types.ModuleType('nltk.data')
    data_mock.path = MockData.path
    data_mock.find = MockData.find
    nltk_mock.data = data_mock
    sys.modules['nltk.data'] = data_mock

    # Set up corpus and its submodules
    corpus_mock = types.ModuleType('nltk.corpus')
    nltk_mock.corpus = corpus_mock
    sys.modules['nltk.corpus'] = corpus_mock

    # Set up stopwords
    stopwords_mock = types.ModuleType('nltk.corpus.stopwords')
    stopwords_mock.words = MockStopwords.words
    corpus_mock.stopwords = stopwords_mock
    sys.modules['nltk.corpus.stopwords'] = stopwords_mock

    # Set up wordnet
    wordnet_mock = types.ModuleType('nltk.corpus.wordnet')
    wordnet_mock.morphy = MockWordNet.morphy
    corpus_mock.wordnet = wordnet_mock
    sys.modules['nltk.corpus.wordnet'] = wordnet_mock

    # Set up stem module and submodules
    stem_mock = types.ModuleType('nltk.stem')
    nltk_mock.stem = stem_mock
    sys.modules['nltk.stem'] = stem_mock

    # Set up wordnet under stem
    stem_wordnet_mock = types.ModuleType('nltk.stem.wordnet')
    stem_wordnet_mock.WordNetLemmatizer = MockWordNetLemmatizer
    stem_mock.wordnet = stem_wordnet_mock
    sys.modules['nltk.stem.wordnet'] = stem_wordnet_mock

    # Create the PorterStemmer
    class MockPorterStemmer:
        def stem(self, word):
            return word

    # Add PorterStemmer directly to stem module
    stem_mock.PorterStemmer = MockPorterStemmer

    # Also add it to the porter submodule
    stem_porter_mock = types.ModuleType('nltk.stem.porter')
    stem_porter_mock.PorterStemmer = MockPorterStemmer
    stem_mock.porter = stem_porter_mock
    sys.modules['nltk.stem.porter'] = stem_porter_mock

    # Also add WordNetLemmatizer directly to stem module
    stem_mock.WordNetLemmatizer = MockWordNetLemmatizer

    # Set up StemmerI API
    stem_api_mock = types.ModuleType('nltk.stem.api')

    class StemmerI:
        """Mock StemmerI interface"""
        pass

    stem_api_mock.StemmerI = StemmerI
    stem_mock.api = stem_api_mock
    sys.modules['nltk.stem.api'] = stem_api_mock

    # Set up translate module
    translate_mock = types.ModuleType('nltk.translate')
    nltk_mock.translate = translate_mock
    sys.modules['nltk.translate'] = translate_mock

    # Set up meteor score
    meteor_mock = types.ModuleType('nltk.translate.meteor_score')
    meteor_mock.meteor_score = lambda x, y: 0.5
    translate_mock.meteor_score = meteor_mock
    sys.modules['nltk.translate.meteor_score'] = meteor_mock

    # Install the main mock
    sys.modules['nltk'] = nltk_mock

    return nltk_mock
