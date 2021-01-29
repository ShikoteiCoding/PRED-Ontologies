import os
import re


class SentenceGenerator(object):  # A generator that returns line of all files in the given path
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file in os.listdir(self.dir_name):
            for line in open(self.dir_name + file, encoding='utf-8'):
                yield line



# for sentence in SentenceGenerator('../Dataset/sentences'):
#     print(sentence)
