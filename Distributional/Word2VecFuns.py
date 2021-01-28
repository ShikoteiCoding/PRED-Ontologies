import re
import gensim
import logging
from bs4 import BeautifulSoup
from typing import List
from nltk.corpus import stopwords
from gensim.models import word2vec, Word2Vec
from Distributional.SentenceGenerator import SentenceGenerator


def sentence_to_wordlist(sentence: str, remove_stopwords=False) -> list[str]:
    """
    return a list of words for the sentence
    :return:
    """
    # remove HTML
    review_text = BeautifulSoup(sentence).get_text()
    # remove non letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    # delete stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words


def corpus_to_sentences(path_to_corpus: str, remove_stopwords=False) -> List[List[str]]:
    """
    get sentences from txt file
    :return: list of words of the sentences
    """
    raw_sentences = SentenceGenerator(path_to_corpus)

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def train_word2vec_model(sentence_folder: str, path_to_model: str, num_workers: int, num_features: int,
                         min_word_count: int,
                         context_size: int, downsampling: float, isSave = True) -> Word2Vec:
    """
    train a word2vec model, if saved the parameters is saved in the same folder
    :param isSave:
    :param num_features:
    :param num_workers:
    :param min_word_count:
    :param downsampling:
    :param context_size:
    :param path_to_model: path where to save the model
    :param sentence_folder:  the folder of the txt file of sentences
    :return:
    """
    sentences = []
    print("Parsing sentences")
    sentences += corpus_to_sentences(sentence_folder)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context_size, sample=downsampling)

    # if don't further train the model this is more memory efficient
    model.init_sims(replace=True)

    # save the model
    if isSave is True:
        model.save(path_to_model)
        with open(path_to_model + '-parameters', 'w') as f_out:
            f_out.write("num_features = %d\nnum_workers = %d\ncontext = %d\ndownsampling = %e"
                        % (num_features, num_workers, context_size, downsampling))
        f_out.close()

    print("finished")
    return model


def load_word2vec_model(model_path: str):
    """
    load a trained word2vec model from its path
    """
    return gensim.models.Word2Vec.load(model_path)


def get_embedding(model, word):
    """
    get a word's vector from a trained model
    :param model:
    :param word:
    :return:
    """
    return model[word]


# model = load_word2vec_model(word2vec_model_folder + '300features_40minwords_10context')
# print(get_embedding(model, 'music'))
# print(get_embedding(model, 'phone'))
