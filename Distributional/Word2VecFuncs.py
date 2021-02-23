from datetime import datetime
from functools import reduce
from typing import List
import spacy

import gensim
import logging
from gensim.models import word2vec, Word2Vec

from Helpers.Generator import TokenGenerator


def train_word2vec_model(sentence_path: str, path_to_model:str, num_workers: int, num_features: int,
                         min_word_count: int,
                         context_size: int, downsampling: float) -> Word2Vec:
    """
    train and return a word2vec model
    :param path_to_model: path where to save the model
    :param sentence_path:  the path of the txt file of sentences, can be a folder or a file
    :return: trained word2vec model
    """

    # model = word2vec.Word2Vec(TokenGenerator(path=sentence_path, keep_stop=True, keep__=True),
    #                           workers=num_workers, size=num_features, min_count=min_word_count,
    #                           window=context_size, sample=downsampling)
    model = word2vec.Word2Vec(TokenGenerator(path=sentence_path, keep__=True), workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context_size, sample=downsampling)

    # forget the original vectors and only keep the normalized ones = saves lots of memory!
    model.wv.init_sims(replace=True)

    # save the model
    model.save(path_to_model)
    return model


def load_word2vec_model(model_path: str):
    """    load a trained word2vec model from its path    """
    return gensim.models.Word2Vec.load(model_path)


def get_embedding(model, word: str):
    # result = reduce(lambda x, y: model.wv[x] + model.wv[y], concept.split()) if len(word.split()) > 1
    try:
        return model.wv[word]
    except Exception as e:
        return None


def get_word_from_embedding(model, embedding):
    return model.wv.most_similar(positive=[embedding])[0][0]


def get_topn_similar(model, word, topn) -> List:
    return model.most_similar(word, topn=topn)


def get_similarity(model, worda, wordb) -> float:
    return model.similarity(worda, wordb)

