import io
import re
from datetime import datetime

import pandas as pd
import spacy
from nltk.corpus import stopwords
sw = stopwords.words("english")
from Helpers import Generator
stop = {'first','two','three','four','five','six', 'second', 'third', 'forth','fifth', 'sixth',
"zero", "one", "two",  "three", "four","five",  "six", "seven", "eight", "nine","ten", "eleven", "twelve", "thirteen", "fourteen",
 "fifteen", "sixteen",  "seventeen", "eighteen", "nineteen" , "twenty", "thirty", "forty", "fifty",
                            "sixty", "seventy", "eighty", "ninety","hundred", "thousand",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "h,erself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "one", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"}
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

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    nlp = spacy.load("en_core_web_sm")
    f = open('../Dataset/sentences/sentence_lemmatized.txt', 'a', encoding='utf-8')
    for i in range(0, 10):
        print(datetime.now(), i)
        for line in open('../Dataset/sentences/sliced_files/sentence_0%d.txt' % i, 'r', encoding='utf-8'):
            line = re.sub("[^Ü-üa-zA-Z_]", " ", line)
            line = re.sub("[.]", " ", line)

            doc = nlp(line)

            sentence = (" ".join([token.lemma_ for token in doc if token.lemma_ not in ['s', 't']]))
            f.write(sentence + '\n')
    f.close()

    # for line in open('../Output/NPs/All NPs mincount = 99.txt', 'r', encoding='utf-8'):
    # words = nlp('previous American Jazz')
    # for w in words.noun_chunks:
    #     print(w.lemma_)
    # sentence = "produced two hit singles including the Top 20 title track and , like I , Assassin , spent six weeks in the charts ."
    # # sentence ="According to Numan , this was an unintentional result of acne ; before an appearance on Top of the Pops , he had `` spots everywhere , so they slapped about half an inch of white makeup on me before I 'd even walked in the door ."
    # words = nlp(sentence)
    #
    # for x in words:
    #     sentence = re.sub(r"\b%s\b" % x.text, x.lemma_, sentence)
    # sentence = re.sub("[^Ü-üa-zA-Z_]", " ", sentence)
    #
    # for chunk in words.noun_chunks:
    #     split = chunk.lemma_.split()
    #     if len(split)>4:
    #         continue
    #     if len(split)>1:
    #         chunk = ' '.join([w for w in split if w not in stop])
    #         res = '_'.join([w for w in split if w not in stop])
    #         sentence = sentence.replace(chunk,res)
    #     # sentence = sentence.replace(x.text, x.lemma_)
    # print(sentence)

    # for sentence in open('../Dataset/sentences/sliced_files/sentence_00.txt', 'r', encoding='utf-8'):
    #     sentence = re.sub("[^Ü-üa-zA-Z_]", " ", sentence)
    #     print(sentence.replace('\n', ''))
    #
    #     doc = nlp(sentence)
    #
    #     for x in doc:
    #         sentence = re.sub(r"\b%s\b" % x.text, x.lemma_, sentence)
    #     sentence = re.sub("[^Ü-üa-zA-Z_]", " ", sentence)
    #     sentence = re.sub(r"\bs\b", '', sentence)
    #
    #     for chunk in doc.noun_chunks:
    #         words = chunk.text.split()
    #         if len(words)>4:
    #             continue
    #         if len(words)>1:
    #             chunk = ' '.join([w for w in words if w not in stop])
    #             res = '_'.join([w for w in words if w not in stop])
    #             sentence = sentence.replace(chunk,res)
    #
    #     print(sentence.split(), '\n')
    # # for token in doc:
    # #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    # #           token.shape_, token.is_alpha, token.is_stop)

    # word = nlp('Music\nArt rock')
# According to Numan , this was an unintentional result of acne ; before an appearance on Top of the Pops , he had `` spots everywhere , so they slapped about half an inch of white makeup on me before I 'd even walked in the door .
# accord to Numan   this be unintentional_result of acne   before appearance on Top of t-PRON- Pops   -PRON- have    spot everyw-PRON-re   so t-PRON-y slap half_inch of white_makeup on -PRON- before -PRON-  d even walk in t-PRON- door
