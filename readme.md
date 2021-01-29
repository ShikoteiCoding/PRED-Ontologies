#Readme 

What is needed to run in the console before running the main.py :
$pip3 install nltk
$pip3 install spacy
$python -m spacy download en_core_web_sm


-Requirements:
    Python 3 (download from Anaconda environment is preferred)
    install nltk
    install spacy

-Download the corpus:
    I provide a link to download the corpus in Corpus/corpus_link

-Download the java module:
    I provide a link to download the java module
    This module should be used to preprocess the corpus

-Dataset (Music.all):
    the Music dataset contains a list of couples, a couple per line (couples are seperated by tab space "\t")
    after the couple there are two more columns: label \t type
    label = True if the couple are hypernym related
    label = False if the couple are not hypernym related
    the type is not useful for your case study (it is used for othey type of semantic relations)
    each couple word is concatenated by its POS tag (e.g: -n refers to noun)

-Sequential patterns (sequential_patterns_Music.txt):
    sequential pattern per line

-Steps to do:
    1- Download the corpus
    2- Download the java module
    3- Preprocess the corpus (corpus_parsing.py)
    4- Build the matrix to learn and test a classifier model using SVM
        rows: dataset couples, columns: sequential patterns, cell value: Extraction count of a couple by the pattern

-Function to read the parsed corpus (it exists in core_functions.py):
    for parsed_sentence in get_sentences(corpus_file): #corpus_file: a path to the parsed corpus
        write the code here to works with the parsed_sentence
        see parsed_sentence.py to know the content of parsed_sentence

-Function to match a pattern with a parsed_sentence (it exists in SP_matching.py):
    spm_matching(pattern, parsed_sentence)
        pattern: a sequential pattern
        parsed_sentence: a parsed sentence
        see more details in SP_matching.py

core_concept = ['music', 'activity', 'arrangement', 'instrument', composition, genre, instrumentation, artist, 'event', 'performance', 'singer', 'recording', 'group', 'record', 'label', 'remix', ]



