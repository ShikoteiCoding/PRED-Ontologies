import base64
import logging
import re
from typing import List

import pandas as pd
from datetime import datetime

from gensim.models.phrases import Phrases, Phraser, pseudocorpus, original_scorer
from Helpers.Generator import stop, LemmaGenerator
from Helpers.Generator import TokenGenerator


# -------------------- Phraser -------------------------------------
def build_phrases_model(sentences, min_count, threshold, progress_per) -> Phraser:
    phrases = Phrases(sentences, min_count=min_count, threshold=threshold, progress_per=progress_per)
    return Phraser(phrases)


def save_phrases_model(phraser: Phraser, path_to_model, train_time) -> Phraser:
    phraser.save(path_to_model)
    with open(path_to_model + '-params.txt', 'w') as f:
        f.write("min_count = %d\n" % phraser.min_count)
        f.write("threshold = %d\n" % phraser.threshold)
        f.write("train time = %s\n" % str(train_time))
        f.write("detected grams = %d\n " % (len(phraser.phrasegrams)))
    return phraser


def load_phrases_model(path) -> Phraser:
    return Phraser.load(path)


def apply_phraser_to_corpus(phraser: Phraser, input_file_path, output_file_path) -> None:
    """
    rebuild corpus file with phrases
    :param phraser:
    :param input_file_path:
    :param output_file_path:
    :return:
    """
    count = 0
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for tokens in TokenGenerator(input_file_path, keep__=True):
            parsed_sentence = ' '.join(phraser[tokens])
            out_file.write(parsed_sentence + '\n')
            count += 1
        print("Added %d sentences" % count)


# ------------------------------- Actions -------------------------------------------------------

def get_all_phrases_and_scores(phraser: Phraser) -> pd.DataFrame:
    """
    get all phrases and their scores calculated by phraser
    :param phraser:
    :return: DataFrame of n-grams and score
    """
    list = []
    for bigram in phraser.phrasegrams.keys():
        list.append([bigram[0].decode('utf-8'), bigram[1].decode('utf-8'), phraser.phrasegrams[bigram]])
    return pd.DataFrame(list, columns=['worda', 'wordb', 'score'])


def get_bigram_score(phraser: Phraser, worda, wordb) -> float:
    """
    Score Score for given bi-gram
    :param phraser:
    :param worda:
    :param wordb:
    :return: if bi-gram not presented in dictionary - return -1.
    """
    return phraser.score_item(worda, wordb, phraser.phrasegrams)


def calculate_score(phrases: Phrases, worda: str, wordb: str) -> str:
    """
    Calculate the score of any given word. Only for Phrases, not applicable for Phraser
    :param phrases:
    :param worda:
    :param wordb:
    :return:
    """
    score = -1
    try:
        score = original_scorer(phrases.vocab[worda.encode('utf-8')], phrases.vocab[wordb.encode('utf-8')],
                                phrases.vocab[(worda+'_'+wordb).encode('utf-8')], len(phrases.vocab),
                                phrases.min_count, phrases.corpus_word_count)
    except Exception as e:
        print(str(e))
        pass
    print("[%s, %s]: %f" % (worda, wordb, score))


def get_model_name(num_of_gram, min_count, threshold):
    return "%d-grams-min%d-threshold%d-file00" % (num_of_gram, min_count, threshold)


def work_phraser(phraser_path, path_to_input, max_gram, min_counts, thresholds) -> Phraser:
    """
    Train a phraser with given corpus and parameters
    'The Emmy Award' -> 'The_Emmy_Award'
    :param folder_of_phraser: Path of folder where stores the phraser
    :param path_to_input: Path of the corpus file or folder
    :param max_gram: Maximum gram allowed. 'The Emmy Award' = 3
    :param min_counts: minimum appearance of a word
    :param thresholds:
    :return: trained phraser
    """
    keep__ = False  # whether to keep the '_' in result; Important because detected phrases are connected with _
    # we don't keep those that are in the original file
    path_to_output = ''
    # start a loop to get a txt file, that combines words in a phrase with _, later use in word2vec model training
    for i in range(2, max_gram + 1):
        start = datetime.now()
        print("start training %d-gram model..." % i)

        # ------------- First step: detect phrases -------------------
        phraser = build_phrases_model(
            LemmaGenerator(path_to_input, keep__=keep__, keep_stop=False),
            min_count=min_counts[i - 2],
            threshold=thresholds[i - 2],
            progress_per=1000)
        end = datetime.now()

        # ------------ Second step: save model and results for analyse --------------------
        model_name = "%d-grams-min%d-threshold%d"  % \
                     (i, min_counts[i - 2], thresholds[i - 2])
        phraser.save(phraser_path + model_name)

        scores = get_all_phrases_and_scores(phraser)
        scores.to_csv(phraser_path + model_name + '_score.csv', index=False)  # save the score

        # ------------- Third step: build a new corpus file with bi-gram phrases -------------------
        path_to_output = phraser_path + model_name + '_sentences.txt'  # used as the input for next loop
        apply_phraser_to_corpus(phraser, path_to_input, path_to_output)
        path_to_input = path_to_output
        keep__ = True

    return phraser


if __name__ == "__main__":
    model = load_phrases_model("Output/Phraser/2-grams-min10-threshold100")
