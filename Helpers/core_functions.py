from nltk import word_tokenize
from nltk import pos_tag
from nltk import WordNetLemmatizer
import spacy
import gzip
from Helpers import parsed_sentence as ps
nlp = spacy.load('en_core_web_sm')
lemma = WordNetLemmatizer()

def get_sentences(corpus_file):
    """
    Returns all the (content) sentences in a processed corpus file
    :param corpus_file: the processed corpus file (may be compressed or not)
    :return: the next sentence (a generator function)
    """
    sent = ps.parsed_sentence()
    # Read all the sentences in the file
    if str(corpus_file).endswith(".gz"):
        f_in = gzip.open(corpus_file, 'r')
    elif str(corpus_file).endswith(".txt"):
        f_in = open(corpus_file, 'r', errors="ignore")
    else:
        print("wrong input file.")
    # with gzip.open(corpus_file, 'r') as f_in:
    isNP = False
    is_root = False
    root = ""
    ri = 0
    np = ""
    np_indexes = []
    for line in f_in:
        # try:
        #     line = str(line, "utf-8")
        # except:
        #     continue
        # Ignore start and end of doc
        if '</s>' in line and sent.id == -1:
            continue
        if '<text' in line or '</text' in line:
            continue
        if '<s id' in line:
            sent.id = line.split("'")[1]
            continue
        # End of sentence
        elif '</s>' in line:
            yield sent
            isNP = False
            is_root = False
            root = ""
            ri = 0
            np = ""
            np_indexes = []
            sent = ps.parsed_sentence()
        elif '<NP>' in line:
            isNP = True
        elif '</NP>' in line:
            isNP = False
            if len(np_indexes) > 0:
                sent.add_NP(np.strip(), root, ri, min(np_indexes), max(np_indexes))
            np = ""
            np_indexes = []
        elif '<root>' in line:
            is_root = True
        elif '</root>' in line:
            is_root = False
        else:
            try:
                word, lemma, pos, index, parent, parent_index, dep, type = line.split("\t")
                if is_root:
                    root = word
                    ri = int(index)
                if isNP:
                    np_indexes.append(int(index))
                    np = np + " " + word
                sent.add_word(word, lemma, pos, int(index), parent, int(parent_index), dep, type.strip())
            # One of the items is a space - ignore this token
            except Exception as e:
                print (str(e))
                continue




def head_and_lemma(couple_term):
    """
    a function that return the lemma of the head word of a noun phrase
    :param couple_term: a term (noun phrase)
    :return: the lemma of the head word of the noun phrase
    """
    if len(str(couple_term).split(" ")) == 1:
        try:
            lem = lemma.lemmatize(couple_term)
        except:
            lem = couple_term
            print ("exception")
        return couple_term, lem
    nn = 0
    text = word_tokenize(couple_term)
    tags = pos_tag(text)
    allTags = [tag[1] for tag in tags]
    ConjFlag = False
    if "CC" in allTags:
        ConjFlag = True
    i = 0
    word = ""
    for tag in tags:
        if str(tag[1]).__eq__("IN") or (ConjFlag and str(tag[1]).__eq__(",")) or (ConjFlag and str(tag[1]).__eq__("CC")):
            break
        if str(tag[1]).__contains__("NN"):
            word = tag[0]
            nn += 1
    try:
        lem = lemma.lemmatize(word)
    except:
        lem = word
        print ("exception")
    return word, lem
