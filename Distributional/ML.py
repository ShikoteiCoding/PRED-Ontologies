import pandas as pd
import os
import json
from gensim.models import Word2Vec
from pandas import DataFrame
from typing import List, Tuple, Dict
from Distributional import Word2VecFuncs
from Distributional.Word2VecFuncs import get_embedding
from Helpers import Generator
from Helpers.HyperHypoCouple import HHCouple, NHHCouple

# Folders

word2vec_model_folder = './Models/'
trial_path = "../Output/trial/"
sentence_folder = '../Dataset/sentences/'


def depreciated_dt_to_embeddings(couples: DataFrame, model: Word2Vec) -> DataFrame:
    """ Get embeddings trained without using Phraser
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


def get_negative_set(path: str) -> List[List[str]]:
    """
    Generate a list of non hypernymy couple based on labeled dataset
    :param path: path of the labeled dataset
    :return: list of non hypernymy couple
    """
    negative_dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split('\t')
        word1, word2, is_hyp = line[0].split('-')[0], line[1].split('-')[0], line[2]
        if is_hyp == "False":
            negative_dataset.append([word1, word2])
    f.close()
    return negative_dataset


def merge_dataset(positive_set: List[List[float]], negative_set: List[List[float]]) -> pd.DataFrame:
    """ # TODO not tested
    return a DataFrame that combines positive and negative hypernymy set
    :param positive_set:
    :param negative_set:
    :return:
    """
    positive = pd.DataFrame(positive_set)
    positive[2] = 'True'
    positive.columns = ['hypo', 'hyper', 'label']

    negative = pd.DataFrame(negative_set)
    negative[2] = 'False'
    return positive.append(negative).drop_duplicates(inplace=True)


def train_model():
    # TODO
    return None


def predict(NP_couples: List[Tuple[str, str]], PCH_Th: float) -> List[Tuple[str, str, float]]:
    # TODO
    result = [('a', 'a', 0)]
    return result


def save_output(result: List[Tuple[str, str, float]], params: Dict, trial_path: str) -> None:
    """ # TODO: need modification to cooperate with new main.py
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



