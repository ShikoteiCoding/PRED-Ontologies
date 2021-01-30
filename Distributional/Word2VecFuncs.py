from datetime import datetime
from functools import reduce
from typing import List

import gensim
import logging
from gensim.models import word2vec, Word2Vec

from Helpers.Generator import TokenGenerator


def train_word2vec_model(sentence_path: str, path_to_model: str, num_workers: int, num_features: int,
                         min_word_count: int,
                         context_size: int, downsampling: float, isSave=True) -> Word2Vec:
    """
    train and return a word2vec model, if saved the parameters are also saved in the same folder
    :param path_to_model: path where to save the model
    :param sentence_path:  the path of the txt file of sentences, can be a folder or a file
    :return: trained word2vec model
    """

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("Training model...")
    start_time = datetime.now()
    model = word2vec.Word2Vec(TokenGenerator(path=sentence_path, keep_stop=True, keep__=True),
                              workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context_size, sample=downsampling)

    # forget the original vectors and only keep the normalized ones = saves lots of memory!
    model.wv.init_sims(replace=True)
    end_time = datetime.now()

    # save the model
    if isSave:
        model.save(path_to_model)
        # save the parameters
        with open(path_to_model + '-parameters.txt', 'w') as f_out:
            f_out.write("num_features = %d\nnum_workers = %d\ncontext = %d\ndownsampling = %e"
                        % (num_features, num_workers, context_size, downsampling))
            f_out.write("train time: "+str(end_time - start_time))

    print("finished")
    return model


def load_word2vec_model(model_path: str):
    """    load a trained word2vec model from its path    """
    return gensim.models.Word2Vec.load(model_path)


def get_embedding(model, concept):
    """    get a single word's vector or added vectors of a concept from the trained model          """
    #            Don't forget that auto-detected phrases are combined with '_' so it would be only one word
    #            the split() is for NPs extracted by patterns
    if len(concept.split() > 1):
        result = reduce(lambda x, y: model.wv[x] + model.wv[y], concept.split())
    else:
        result = model.wv[concept]
    return result


def get_topn_similar(model, word, topn) -> List:
    return model.most_similar(word, topn=topn)


def get_similarity(model, worda, wordb) -> float:
    return model.similarity(worda, wordb)


