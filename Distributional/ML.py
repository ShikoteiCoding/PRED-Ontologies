import gc
import math

import numpy as np
import pandas as pd
import os
import json
from gensim.models import Word2Vec
from pandas import DataFrame
from typing import List, Tuple, Dict, Set

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from Distributional import Word2VecFuncs
from Distributional.Word2VecFuncs import get_embedding
from Helpers import Generator
from Helpers.HyperHypoCouple import HHCouple, NHHCouple
from sklearn.model_selection import train_test_split

# Folders

word2vec_model_folder = './Models/'
trial_path = "../Output/trial/"
sentence_folder = '../Dataset/sentences/'


def boost_embeddings(data: DataFrame, n: int) -> DataFrame:
    """
    Boost the positive dataset by repeating it n times
    :param data:
    :param n:
    :return:
    """
    result = data
    for i in range(2, n + 1):
        result = result.append(data)
    print(len(result))
    return result

    # def get_hhcouple_embeddings(couples: DataFrame, model: Word2Vec) -> DataFrame:
    #     """ Get embeddings trained without using Phraser
    #     :param couples:
    #     :param model:
    #     :return:
    #     """
    #     couples.loc[:, 'hypo'] = couples['hypo'].map(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x)
    #     couples.loc[:, 'hyper'] = couples['hyper'].map(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x)
    #
    #     dt_embeddings = couples.applymap(lambda x: get_embedding(model, x))
    #     # dt_embeddings = couples.applymap(lambda x: get_embedding(model,
    #     #                                                               str(lambda y: '_'.join(y.split())if len(y.split() )> 1 else y)))
    #     # print("drop %d lines：" % sum(dt_embeddings.shape[1] - dt_embeddings.count(axis=1)))
    #     dt_embeddings.dropna(inplace=True)
    #     return dt_embeddings


def get_embeddings_from_txt(path: str, w2v, hhcouples: DataFrame) -> DataFrame:
    """
    Get the word embeddings from a labeled text
    :param path: path of the labeled dataset
    :return: list of non hypernymy couple
    """
    nhhcouples = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split('\t')
        word1, word2, is_hyp = line[0].split('-')[0], line[1].split('-')[0], line[2]
        if is_hyp == 'False':
            if word1 in hhcouples.hypo.values or word2 in hhcouples.hypo.values or word1 in hhcouples.hyper.values \
                    or word2 in hhcouples.hyper.values:
                nhhcouples.append([word1, word2])
    embeddings = return_features_from_word(pd.DataFrame(nhhcouples), w2v)
    return embeddings


def merge_dataset(positive: DataFrame, negative: DataFrame) -> pd.DataFrame:
    """return a DataFrame that combines positive and negative hypernymy set
    colomn[0] = worda, column[1] = wordb, column[2] = label
    :return:
    """
    positive['label'] = 1
    negative['label'] = 0
    return positive.append(negative)


def return_features_from_word(data: DataFrame, w2v, limit=None) -> DataFrame:
    """ Given two words, return a dataframe with their embeddings"""
    if limit is not None:
        data = data.loc[:, limit]

    data = data.applymap(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x)
    data = data.applymap(lambda x: get_embedding(w2v, x))
    data.columns = ['vec_a', 'vec_b']

    data.dropna(inplace=True)

    return data


def self_concat(dataset: DataFrame):
    """
    return a 2d+1 DataFrame as input features
    :param dataset:
    :return:
    """
    combine = pd.DataFrame(np.column_stack(list(zip(*dataset['vec_a'])) + list(zip(*dataset['vec_b'])))).reset_index(drop=True)
    lastcol = (dataset['vec_a']-dataset['vec_b'])\
        .apply(lambda x: np.linalg.norm(x, ord=1)).reset_index(drop=True)
    result = pd.concat([combine, lastcol], axis=1)

    return result


def train_model(dataset: DataFrame, show_cross_val=True):
    X_dataset = self_concat(dataset)
    y_dataset = dataset['label']
    # clf = AdaBoostClassifier(svm.SVC(probability=True), n_estimators=10, learning_rate=1.0)

    clf = svm.SVC(probability=True)
    if show_cross_val:
        cross_val_score(clf, X_dataset, y_dataset, cv=5, scoring=make_scorer(classification_report_with_f1_score))

    clf_res = clf.fit(X_dataset, y_dataset)
    return clf_res


def classification_report_with_f1_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return precision_score(y_true, y_pred)


def get_predict_result(nps: DataFrame, clf, isSaved=False, path_predict=None) -> DataFrame:
    predict = clf.predict_proba(self_concat(nps))
    nps['y_prob_1'] = predict[:, 1].round(2)
    nps.sort_values('y_prob_1', inplace=True, ascending=False)
    nps.drop(columns=['vec_a', 'vec_b'], inplace=True)
    if isSaved:
        nps.to_csv(path_predict)
    return nps


def save_predict_set(nps: DataFrame, model, path):
    """ Save the NP pair embeddings in csv, to be run only once to build the massive matrix """
    count = 0
    step = 100

    for i in range(0, int(nps.shape[0] / 100)):
        dfa = nps[i * step: (i + 1) * step]
        dfa.loc[:, ['key']] = 0
        dfb = dfa
        merged = pd.merge(dfa, dfb, on='key').drop(columns=['key'], axis=1)
        embeddings = merged.applymap(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x).applymap(
            lambda x: get_embedding(model, x))
        result = pd.concat([merged, embeddings], axis=1)
        result.columns = ['NP_a', 'NP_b', 'vec_a', 'vec_b']
        result.dropna(axis=0, inplace=True)
        result.to_pickle("%s%d.pkl" % (path, i))  # append mode
        count = i
        del dfa, dfb, merged

    dfa = nps[count * step + 1:]
    dfa['key'] = 0
    dfb = dfa
    merged = pd.merge(dfa, dfb, on='key').drop(columns=['key'], axis=1)
    embeddings = merged.applymap(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x).applymap(
        lambda x: get_embedding(model, x))
    result = pd.concat([merged, embeddings], axis=1)
    result.columns = ['NP_a', 'NP_b', 'vec_a', 'vec_b']
    result.dropna(axis=0, inplace=True)
    result.to_pickle("%s%d.pkl" % (path, count+1))  # append mode


def build_predict_embedding_pairs(nps: DataFrame, model, limit) -> DataFrame:
    """ Build the predict set from filtered NPs """
    nps['key'] = 0
    dt = nps.loc[:limit]
    result = pd.merge(dt, nps.loc[:limit], on='key').drop(columns=['key'], axis=1)
    embeddings = result.applymap(lambda x: '_'.join(x.split()) if len(x.split()) > 1 else x).applymap(lambda x: get_embedding(model, x))
    result = pd.concat([result, embeddings], axis=1)
    result.columns = ['NPa', 'NPb', 'worda', 'wordb']
    result.dropna(axis=0, inplace=True)

    # result.columns=['worda_name', 'wordb_name', 'worda', 'wordb']

    result.to_csv('Output/test/predict embedding pairs.csv')
    return result

def converter(instr):
    return np.fromstring(instr[1:-1], sep=' ')


def load_predict_embedding_pairs(path_to_predict_pair, limit=None) -> DataFrame:
    # TODO: read in chunk if it's too big

    pairs = pd.concat([pd.read_pickle(path_to_predict_pair + x) for x in os.listdir(path_to_predict_pair)])
    return pairs



def save_output(result: DataFrame,  trial_path: str, params=None) -> None:
    """
    Save the output after one iteration; couples in .csv, parameters in .txt
    :param params: hyper parameters for the model
    :param result: the predict result of the model
    :param trial_path: path of the folder (one level above each iteration)
    :return:
    """
    if not os.path.exists(trial_path):
        try:
            os.makedirs(trial_path)
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

    if params is not None:
        with open(folder_path + 'params.txt', 'w') as f_out:
            # f_out.write(json.dumps(params))
            json.dump(params, f_out)


def parse_output(trial_path, iteration) -> (DataFrame):
    for file in os.listdir(_get_iter_folder_path(trial_path, iteration)):
        if file.endswith("output.csv"):
            csv_file = _get_iter_folder_path(trial_path, iteration) + file
            data = pd.read_csv(csv_file)
    return data


def _get_iter_folder_path(path, iteration) -> str:
    return path + 'iter_' + str(iteration) + '/'


def converter(instr):
    return np.fromstring(instr[1:-1], sep=' ')

if __name__ == '__main__':
    df = pd.DataFrame({'worda': ['band'],
                       'wordb':['new'],
                       'embed1': [np.array([1, 2,3])],
                        'embed2': [np.array([3, 4,5])]})
    df.to_csv('tmp.csv')
    df3 = pd.read_csv('tmp.csv')
    print(df3)
    df1 = pd.read_csv('tmp.csv', converters={'embed1': converter, 'embed2': converter})
    print(df)

    print(len(df1.iloc[0, 2]))
