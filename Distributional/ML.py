import pandas as pd
import os
import json
from gensim.models import Word2Vec
from pandas import DataFrame
from typing import List, Tuple, Dict

from Distributional import Word2VecFuns
from Distributional.Word2VecFuns import get_embedding

# Folders

word2vec_model_folder = './Models/'
trial_path = "../Output/trial/"
sentence_folder = '../Dataset/sentences/'


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
    dt_embeddings = pd.DataFrame()
    # TODO
    return dt_embeddings


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


# test_list = [('aa', 'bb', '0.01'), ('wh', 'hw', '9951')]
# test_dict = '{ "name":"John", "age":30, "city":"New York"}'
#
#
# save_output(test_list, test_dict, trial_path)
# data, params = parse_output(trial_path, 3)
# print(data)
# print(params)

model = create_word2vec_model(sentence_folder)
embeddings = get_embedding(model, 'music')
print(embeddings)
