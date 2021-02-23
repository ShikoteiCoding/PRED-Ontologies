#Readme 

What is needed to run in the console before running the main.py :
$pip3 install nltk
$pip3 install spacy
$python -m spacy download en_core_web_sm


- Requirements:
    Python 3 (download from Anaconda environment is preferred)
    install nltk
    install spacy

- Download the corpus:
    I provide a link to download the corpus in Corpus/corpus_link

- Download the java module:
    I provide a link to download the java module
    This module should be used to preprocess the corpus

###  Dataset (Music.all):
   the Music dataset contains a list of couples, a couple per line (couples are seperated by tab space "\t")
    after the couple there are two more columns: label \t type
    label = True if the couple are hypernym related
    label = False if the couple are not hypernym related
    the type is not useful for your case study (it is used for othey type of semantic relations)
    each couple word is concatenated by its POS tag (e.g: -n refers to noun)

### Sequential patterns (sequential_patterns_Music.txt):
sequential pattern per line


### Function to read the parsed corpus (it exists in core_functions.py):
   4 functions to get sentences for couple extraction and NP extraction from a file of a folder

### MLFuncs.py
Contains all the functions to support the classifier
- Get embeddings
- Construct training dataset
- Construct predict NP set
- Train the model with svm or xgboost
- Evaluate the model
- Get predict result

### PhraserFuncs.py
Contains all the functions related to word2vec models
- Train a phraser
- Get score of phrases caculated by the phraser
- Apply phraser to the original corpus to build a new corpus with phrases

### Word2VecFuncs.py
Contains all the functions related to word2vec models
- Train a word2vec model
- Get similarities with the model

### PatternFuncs.py
Contains all the functions related to hypernym couple and NP extraction


### Function to match a pattern with a parsed_sentence (it exists in SP_matching.py):
spm_matching(pattern, parsed_sentence
- pattern: a sequential pattern
-  parsed_sentence: a parsed sentence
- see more details in SP_matching.py

### main.py
The main program. 
1. Redefine the paths if needed
2. For time consuming processes (building word2vec model, predict NP set ...) the program checks if the file path exists first.
3. Detailed steps are in comments
4. The condition of existing the iteration loop is currently set to comparing newly discovered hyponyms with the hypernym set;



