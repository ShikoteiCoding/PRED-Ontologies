import logging
from datetime import datetime
from typing import Generator

import gensim
from gensim.models.phrases import Phrases, Phraser

from Distributional.SentenceGenerator import SentenceGenerator
from Distributional.Word2VecFuns import sentence_to_wordlist


# -------------------- Phraser -------------------------------------
def build_phrases(sentences, min_count, threshold, progress_per) -> Phraser:
    phrases = Phrases(sentences,
                      min_count=min_count,
                      threshold=threshold,
                      progress_per=progress_per)

    return Phraser(phrases)


def save_phrases_model(phraser: Phraser, model_path, model_name, train_time) -> None:
    phraser.save(model_path + model_name + '.txt')

    with open(model_path + model_name + '-params.txt', 'w') as f:
        f.write("min_count = %d" % phraser.min_count)
        f.write("threshold = %d " % phraser.threshold)
        f.write("train time = " + str(train_time))


def load_phrases_model(path) -> Phraser:
    return Phraser.load(path)


# ----------------------- rebuild corpus file with phrases --------------------------------------
def _sentence_to_bi_grams(phrases_model, sentence):
    return ' '.join(phrases_model[sentence])


def sentences_to_bi_grams(n_grams, input_file_folder, output_file_name):
    count = 0
    with open(output_file_name, 'w+', encoding='utf-8') as out_file:
        for sentence in SentenceGenerator(input_file_folder):
            tokenized_sentence = sentence_to_wordlist(sentence)
            parsed_sentence = _sentence_to_bi_grams(n_grams, tokenized_sentence)
            out_file.write(parsed_sentence + '\n')
            count += 1
            if count % 10000 == 0:
                print(count)


# ------------------------------- Actions -------------------------------------------------------

def do_sentences_to_bigrams():
    """
    Second step: build a new corpus file with bi-gram phrases
    :return:
    """
    path_to_model = 'Phrasers/bi-grams1.txt'
    path_to_sentence_folder = '../Dataset/sentences/sliced_files/'
    path_to_bigram_sentence = '../Dataset/sentences/2grams_sentences.txt'
    model = load_phrases_model(path_to_model)
    sentences_to_bi_grams(model, path_to_sentence_folder, path_to_bigram_sentence)


def do_bigrams_model():
    """
    First step: build bi-gram phrases based on corpus sentences
    :return:
    """
    path_to_sentence_folder = '../Dataset/sentences/sliced_files/'
    path_to_model = 'Phrasers/'
    model_name = 'bi-grams1'
    start = datetime.now()
    print("start training...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    phraser = build_phrases(SentenceGenerator(path_to_sentence_folder),
                            min_count=5,
                            threshold=10,
                            progress_per=10000)
    end = datetime.now()
    print("finished")
    save_phrases_model(phraser, path_to_model, model_name, end - start)


# --------------------------------------- run -------------------------------------------------------
do_sentences_to_bigrams()
