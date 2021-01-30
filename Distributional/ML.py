from functools import reduce

import numpy as np
import pandas as pd
import os
import json
from gensim.models import Word2Vec
from pandas import DataFrame
from typing import List, Tuple, Dict, Set
from Distributional import Word2VecFuns
from Distributional.Word2VecFuns import get_embedding
from Helpers import core_functions as cf
from nltk.corpus import stopwords

# Folders

word2vec_model_folder = './Models/'
trial_path = "../Output/trial/"
sentence_folder = '../Dataset/sentences/'


def test():
    return None
    # filter_NP('../Dataset/NPs/all_NPs.txt')
    # model = create_word2vec_model('../Dataset/sentences/sliced_files/', 'without_number')



def lower_NPs_refilter(path):
    result = []
    count = 0
    with open(path, 'r') as f:
        NPs = json.load(f)
    for couple in NPs:
        isSave = True
        if '.' in couple:  # compilation.The cons
            couple = couple[:couple.index('.')]  # compilation
        if couple.isupper(): # MY LOINS TREMBLE
            continue
        if not couple.istitle():
            couple = couple.lower()
        for word in couple.split():
            if word.isdigit() or word in get_stopwords():
                isSave = False
                break
        if isSave:
            count += 1
            result.append(couple)
        if count % 1000 ==0:
            print(count)
    print(count)
    with open(path.replace('.txt', '-refiltered.txt'), 'w') as fout:
        json.dump(list(set(result)), fout)




def get_stopwords() -> Set:
    result = set()
    stop = cf.stopWords
    stop.add("The")
    stop.add("This")
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
    stop.add("one")
    stop.add(x for x in stopwords.words("english"))

    return result

def create_word2vec_model(sentence_folder, model_name=None, isSave=True) -> Word2Vec:  # hyper parameter
    """
    return the word2vec model
    :param isSave: whether to save the model
    :param sentence_folder: folder where the sentence file is stored (_tokenized.txt)
    :param model_name: specify the model name, if not it can be generated automatically after
    :return:
    """
    # setting hyper parameters
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context_size = 10  # context window size
    downsampling = 1e-3  # for frequent words

    if model_name is None:
        model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context_size)
    path_to_model = word2vec_model_folder + model_name

    return Word2VecFuns.train_word2vec_model(sentence_folder, path_to_model,
                                             num_workers, num_features, min_word_count, context_size, downsampling, isSave)


def dt_to_embeddings(couples: DataFrame, model: Word2Vec) -> DataFrame:
    """

    :param couples:
    :param model:
    :return:
    """
    # couples = pd.read_csv('../Dataset/first_dataset.csv', index_col ='index',  names=['index', 'hypo', 'hyper', 'label', 'pattern'])
    dt_embeddings = pd.DataFrame(columns=['hypo', 'hyper', 'label'])
    for index, line in couples.iterrows():
        try:
            hypo = get_embedding(model, line['hypo'])
            hyper = get_embedding(model, line['hyper'])
            label = line['label']
            dt_embeddings.loc[index] = [hypo, hyper, label]
        except Exception as e:
            # word not in vocabulary
            continue
    return dt_embeddings


def filter_NP(path_to_NP, model: Word2Vec, STOPWORDS): #TODO: rerun this after getting 00_NPs
    """
    filter the NPs that are not in the vocabulary of the model
    """
    saved_NPs = []
    with open(path_to_NP, "r") as f:
        origin_NPs = json.load(f)
    for NP in origin_NPs:
        is_save = False
        if '.' in NP:  # compilation.The cons
            NP = NP[:NP.index('.')]  # compilation
        if NP.isupper():  # MY LOINS TREMBLE
            continue
        if not NP.istitle():
            NP = NP.lower()
        words = NP.split()
        if len(words) > 3:
            continue
        for word in words:
            if word.isdigit() or word in STOPWORDS:
                is_save = False
                break
            if word in model.wv:
                is_save = True
        if is_save:
            saved_NPs.append(NP)
    saved_NPs = list(set(saved_NPs))
    with open(path_to_NP.replace('.txt', '-filtered.txt'), 'w') as f_out:
        json.dump(saved_NPs, f_out)
    return saved_NPs


def train_model():
    # TODO
    return None


def predict(NP_couples: List[Tuple[str, str]], PCH_Th: float) -> List[Tuple[str, str, float]]:
    # TODO
    result = [('a', 'b', 0)]
    return result


def save_output(result: List[Tuple[str, str, float]], params: Dict, trial_path: str) -> None:
    """
    Save the output after one iteration; couples in .csv, parameters in .txt
    :param params: hyper parameters for the model
    :param result: the predict result of the model
    :param trial_path: path of the folder (one level above each iteration)
    :return:
    """
    result = pd.DataFrame(result)
    if not os.path.exists(trial_path):
        try:
            os.mkdir(trial_path)
        except OSError as e:
            print("create dir failed", str(e))
    dir_count = len(os.listdir(trial_path))
    folder_path = _get_iter_folder_path(trial_path, dir_count + 1)
    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
        except OSError:
            print("create dir failed")
    result.to_csv(folder_path + 'output.csv')

    with open(folder_path + 'params.txt', 'w') as f_out:
        # f_out.write(json.dumps(params))
        json.dump(params, f_out)
    f_out.close()


def parse_output(trial_path, iteration) -> (DataFrame, Dict):
    for file in os.listdir(_get_iter_folder_path(trial_path, iteration)):
        if file.endswith(".csv"):
            csv_file = _get_iter_folder_path(trial_path, iteration) + file
            data = pd.read_csv(csv_file)
        elif file.endswith(".txt"):
            txt_file = _get_iter_folder_path(trial_path, iteration) + file
            with open(txt_file) as f_json:
                params = json.load(f_json)

    return data, params


def _get_iter_folder_path(path, iteration) -> str:
    return path + 'iter_' + str(iteration) + '/'



