import os
import re
from spacy.lang.en.stop_words import STOP_WORDS

stop = set(STOP_WORDS)
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


class BasicGenerator(object):
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


class LineGenerator(object):  # A generator that returns lines of all files in the given path
    def __init__(self, dir_name, keep__=True, keep_stop=True):
        self.path = dir_name
        self.keep_stop = keep_stop  # whether to keep the stop word in result
        self.keep__ = keep__

    def __iter__(self):
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                for line in open(self.path + file, encoding='utf-8', errors="ignore"):
                    yield return_res(line, return_line=True, keep_stop=self.keep_stop, keep__=self.keep__)
        else:
            for line in open(self.path, encoding='utf-8', errors="ignore"):
                yield return_res(line, return_line=True, keep_stop=self.keep_stop, keep__=self.keep__)


class TokenGenerator(object):  # A generator that returns list of words of all files in the given path,
    # filtering the stop words
    def __init__(self, path, keep__=True, keep_stop=True):
        self.path = path
        self.keep__ = keep__  # whether to keep the '_' sign in return
        self.keep_stop = keep_stop  # whether to keep the stop word in result

    def __iter__(self):
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                for line in open(self.path + file, encoding='utf-8', errors="ignore"):
                    yield return_res(line, return_line=False, keep__=self.keep__, keep_stop=self.keep_stop)

        else:
            for line in open(self.path, encoding='utf-8', errors="ignore"):
                yield return_res(line, return_line=False, keep__=self.keep__, keep_stop=self.keep_stop)


def return_res(line, return_line, keep__=True, keep_stop=True):
    line = re.sub("[^Ü-üa-zA-Z-_]", " ", line)
    line = re.sub("[.]", " ", line)
    if not keep__:  # remove the original _ character to not interfere with Phraser
        line = re.sub("[_]", " ", line)
    list = line.split()
    #     list[0] = list[0].lower()
    if not keep_stop:
        res = [w for w in list if w not in stop]
        return " ".join(res) if return_line else res

    else:
        return " ".join(list) if return_line else list


if __name__ == "__main__":
    for word in stop:
        print(word)