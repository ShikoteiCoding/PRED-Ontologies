import os
import re
from spacy.lang.en.stop_words import STOP_WORDS

stop = set(STOP_WORDS)
stop.add("The")
stop.add("This")
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


class LineGenerator(object):  # A generator that returns lines of all files in the given path
    def __init__(self, dir_name, is_dir=True, keep_stop=True):
        self.dir_name = dir_name
        self.is_dir = is_dir
        self.keep_stop = keep_stop  # whether to keep the stop word in result

    def __iter__(self):
        if self.is_dir:
            for file in os.listdir(self.dir_name):
                for line in open(self.dir_name + file, encoding='utf-8'):
                    yield return_res(line, is_line=True)
        else:
            for line in open(self.dir_name, encoding='utf-8'):
                yield return_res(line, is_line=True)


class TokenGenerator(object):  # A generator that returns list of words of all files in the given path,
    # filtering the stop words
    def __init__(self, dir_name, is_dir=True, keep__=False, keep_stop=True):
        self.dir_name = dir_name
        self.is_dir = is_dir
        self.keep__ = keep__  # whether to keep the '_' sign in return
        self.keep_stop = keep_stop  # whether to keep the stop word in result

    def __iter__(self):
        if self.is_dir:
            for file in os.listdir(self.dir_name):
                for line in open(self.dir_name + file, encoding='utf-8'):
                    yield return_res(line, is_line=False, keep__=self.keep__, keep_stop=self.keep_stop)

        else:
            for line in open(self.dir_name, encoding='utf-8'):
                yield return_res(line, is_line=False, keep__=self.keep__, keep_stop=self.keep_stop)


def return_res(line, is_line, keep__=True, keep_stop=True):
    line = re.sub("[^Ü-üa-zA-Z-_]", " ", line)
    line = re.sub("[.]", " ", line)
    if not keep__:  # remove the original _ character to not interfere with Phraser
        line = re.sub("[_]", " ", line)
    list = line.split()
    if len(list) > 1 and list[0].istitle() and not list[1].istitle():  # lowercase for the first letter of the sentence
        list[0] = list[0].lower()
    if not keep_stop:
        res = [w for w in list if w not in stop]
        return " ".join(res) if is_line else res

    else:
        return " ".join(list) if is_line else list
