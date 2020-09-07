import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import unicodedata
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import six

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens  


### Clean our Text and return a list of tokens ###
class PlainTokenizer(object):
    """
        Runs basic tokenization (punctuation splitting, lower casing, etc.).
        Ner terms aren't used in this tokenizer. 
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    
    def tokenize(self, text):
        """Tokenizes a piece of text."""

        # convert 'text' to unicode 
        text = convert_to_unicode(text)

        # remove invalide char and white space 
        text = self._clean_text(text)

        # spliting on a piece of text 
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()

                # remove accent (Nonspacing Mark from the tokens)
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class NERTokenizer(PlainTokenizer):

    """
        Subclass of PlainTokenizer and override 'tokenize' function 
        to keep the ner terms intact
    """

    def __init__(self, do_lower_case=True):
        super().__init__(do_lower_case=True)

    def tokenize(self, text, terms):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []

        for token in orig_tokens:

            # remain terms untouched
            if token in terms:
                split_tokens.extend([token])
                continue

            # if the token contains multiple (>=1) terms
            elif sum([term in token for term in terms]) > 0:
                for term in terms:
                    if term in token:
                        splited = token.split(term)

                        # result: token_0 + term + token_1
                        token = splited[0] #token_0
                        if self.do_lower_case:
                            token = token.lower()
                            token = self._run_strip_accents(token)
                        split_tokens.extend(self._run_split_on_punc(token))
                        split_tokens.append(term) # term
                        token = splited[1] # token_1
                        if self.do_lower_case:
                            token = token.lower()
                            token = self._run_strip_accents(token)
                        split_tokens.extend(self._run_split_on_punc(token))
                continue

            # if text is not in terms 
            else:
                if self.do_lower_case:
                    token = token.lower()
                    token = self._run_strip_accents(token)
                split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
