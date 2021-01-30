import base64
import logging
import re
import pandas as pd
from datetime import datetime

from gensim.models.phrases import Phrases, Phraser, pseudocorpus
from Helpers.Generator import stop
from Helpers.Generator import TokenGenerator


def clean_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r'[^a-z0-9A-Z\s]', ' ', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)


def tokenize(sentence):
    return [token for token in sentence.split() if token not in stop]
    # return " ".join(word for word in list)


# -------------------- Phraser -------------------------------------
def build_phrases_model(sentences, min_count, threshold, progress_per) -> Phrases:
    phrases = Phrases(sentences, min_count=min_count, threshold=threshold, progress_per=progress_per)
    return phrases


def save_phrases_model(phrases: Phrases, path_to_model, train_time) -> Phraser:
    all_phrases = report_all_phrases(phrases)
    df = pd.DataFrame(all_phrases)
    df.to_csv(path_to_model + '-phrases.csv')  # save all phrases calculated
    phraser = Phraser(phrases)
    phraser.save(path_to_model)
    with open(path_to_model + '-params.txt', 'w') as f:
        f.write("min_count = %d\n" % phrases.min_count)
        f.write("threshold = %d\n" % phrases.threshold)
        f.write("train time = %s\n" % str(train_time))
        f.write("detected grams = %d\n " % (len(phraser.phrasegrams)))
    return phraser


def load_phrases_model(path) -> Phraser:
    return Phraser.load(path)


# ----------------------- rebuild corpus file with phrases --------------------------------------

def sentences_to_bi_grams(phraser: Phraser, input_file_path, output_file_path):
    count = 0
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for tokens in TokenGenerator(input_file_path, keep__=True):
            parsed_sentence = ' '.join(phraser[tokens])
            out_file.write(parsed_sentence + '\n')
            count += 1
            if count % 10000 == 0:
                print("Adding phrases to %d sentences" % count)
        print("Added %d sentences" % count)


# ------------------------------- Actions -------------------------------------------------------

def get_bigram_score(model: Phraser) -> pd.DataFrame:
    """ return a list of bigram and its score"""
    list = []
    for bigram in model.phrasegrams.keys():
        list.append([bigram[0].decode('utf-8'), bigram[1].decode('utf-8'), model.phrasegrams[bigram]])
    return pd.DataFrame(list)


# def print_known_gram_score(phrases: Phrases, worda:str, wordb: str, bigram: str) -> str:
#     score = original_scorer(phrases.vocab[worda.encode('utf-8')], phrases.vocab[wordb.encode('utf-8')],
#                     phrases.vocab[bigram.encode('utf-8')], len(phrases.vocab),
#                     phrases.min_count, phrases.corpus_word_count)
#     print("[%s, %s]: %f" % (worda, wordb, score))
#     phrases.score_item()


def _get_model_name(num_of_gram, min_count, threshold):
    return "%d-grams-min%d-threshold%d-file00" % (num_of_gram, min_count, threshold)


def report_all_phrases(phrases_model):
    corpus = pseudocorpus(phrases_model.vocab, phrases_model.delimiter, phrases_model.common_terms)
    phrasegrams = set()
    for bigram, score in phrases_model.export_phrases(corpus, as_tuples=True):
        phrasegrams.add(bigram)
    return phrasegrams


def work_phraser(folder_of_phraser, path_to_input, folder_of_new_corpus, max_gram, min_counts, thresholds) -> Phraser:
    """ Work the phraser with given parameters
    :return The path of phrased sentences"""
    keep__ = False  # whether to keep the '_' in result; important because detected phrases are connected with _
                    # we don't keep it that is in the original file
    path_to_output = ''
    # start a loop to get a txt file, that combines words in a phrase with _, later use in word2vec model training
    for i in range(2, max_gram + 1):
        start = datetime.now()
        print("start training %d-gram model..." % i)

        # ------------- First step: detect phrases -------------------
        phrases = build_phrases_model(
            TokenGenerator(path_to_input, keep__=keep__, keep_stop=False),
            min_count=min_counts[i - 2],
            threshold=thresholds[i - 2],
            progress_per=1000)
        end = datetime.now()

        # ------------ Second step: save model and results for analyse --------------------
        model_name = "%d-grams-min%d-threshold%d" % \
                     (i, min_counts[i - 2], thresholds[i - 2])
        phraser = save_phrases_model(phrases, folder_of_phraser + model_name, end - start)  # save the model and all calculated pairs
        get_bigram_score(phraser).to_csv(folder_of_phraser + model_name + '_score.csv', index=False)  # save the score

        # ------------- Third step: build a new corpus file with bi-gram phrases -------------------
        path_to_output = folder_of_new_corpus + model_name + '.txt'  # used as the input for next loop
        sentences_to_bi_grams(phraser, path_to_input, path_to_output)
        path_to_input = path_to_output
        keep__ = True

    return path_to_output


if __name__ == "__main__":
    # for test purpose
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # enable logging

    original_file_name = 'sentence_00.txt'
    # path
    path_to_sentence_folder = '../Dataset/sentences/sentence_00/'
    path_to_model_folder = 'Phrasers/'
    path_to_input_sentence = path_to_sentence_folder + original_file_name

    max_gram = 4  # the maximum gram count
    min_counts = [30, 20, 20]  # TODO test params, to be tuned with the score.csv
    thresholds = [100, 20, 10]  # only phrases above the threshold will be combined as "x_y" in output file

    keep__ = False  # whether to keep the '_' in result; important because detected phrases are connected with _
    # we don't keep it that is in the original file

    # start a loop to get a txt file, that combines words in a phrase with _, later use in word2vec model training
    for i in range(2, max_gram + 1):
        start = datetime.now()
        print("start training %d-gram model..." % i)

        # ------------- First step: detect phrases -------------------
        phraser = build_phrases_model(
            TokenGenerator(path_to_input_sentence, keep__=keep__, keep_stop=False),
            min_count=min_counts[i - 2],
            threshold=thresholds[i - 2],
            progress_per=1000)
        end = datetime.now()

        # ------------ Second step: save model and results for analyse --------------------
        model_name = "%d-grams-min%d-threshold%d" % \
                     (i, min_counts[i - 2], thresholds[i - 2])
        save_phrases_model(phraser, path_to_model_folder + model_name, end - start)  # save the model
        get_bigram_score(phraser).to_csv(path_to_model_folder + model_name + '_score.csv')  # save the score

        # ------------- Third step: build a new corpus file with bi-gram phrases -------------------
        path_to_output_sentence = path_to_sentence_folder + model_name + '.txt'  # used as the input for next loop
        sentences_to_bi_grams(phraser, path_to_input_sentence, path_to_output_sentence)
        path_to_input_sentence = path_to_output_sentence
        keep__ = True
