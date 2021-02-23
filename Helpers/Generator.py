import os
import re

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

stop = set(STOP_WORDS)
stop.add('PRON')
stop.add("The")
stop.add("This")
stop.add("It")
stop.add("to")
stop.add("seven")
stop.add("especially")
stop.add("other")
stop.add("major")
stop.add("numerous")
stop.add("different")
stop.add("new")
stop.add("newer")
stop.add("primary")
stop.add("previous")
stop.add("slower")
stop.add("similar")
stop.add("recent")
stop.add("later")
stop.add("better")
stop.add("biggest")
stop.add("good")
stop.add("one")


class LineGenerator(object):
    """ Generator of lines in a file or all files in a folder without modification"""
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                for line in open(self.path + file, encoding='utf-8', errors='ignore'):
                    yield line
        else:
            for line in open(self.path, encoding='utf-8', errors="ignore"):
                yield line


class TokenGenerator(object):
    """ Generator of tokens in a file or all files in a folder without modification"""
    def __init__(self, path, keep__):
        self.path = path
        self.keep__ = keep__

    def __iter__(self):
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                for line in open(self.path + file, encoding='utf-8', errors='ignore'):
                    if not self.keep__:
                        line = re.sub("[_]", " ", line)
                    yield line.split()
        else:
            for line in open(self.path, encoding='utf-8', errors="ignore"):
                if not self.keep__:
                    line = re.sub("[_]", " ", line)
                yield line.split()


class LemmaGenerator(object):  # A generator that returns list of words lemmatized by spacy
    # filtering the stop words
    def __init__(self, path, keep__=True, keep_stop=True):
        self.path = path
        self.keep__ = keep__  # whether to keep the '_' sign in return
        self.keep_stop = keep_stop  # whether to keep the stop word in result
        self.nlp = spacy.load("en_core_web_sm")

    def __iter__(self):
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                for line in open(self.path + file, encoding='utf-8', errors="ignore"):
                    yield return_lemmatized_tokens(self, line)

        else:
            for line in open(self.path, encoding='utf-8', errors="ignore"):
                yield return_lemmatized_tokens(self, line)


def return_lemmatized_tokens(self, sentence):
    line = re.sub("[^Ü-üa-zA-Z_0-9]", " ", sentence)
    line = re.sub("[.]", " ", line)

    doc = self.nlp(line)
    return [token.lemma_ for token in doc if token.lemma_ not in ['s', 't']]
