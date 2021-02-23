
import xgboost as xgb
from itertools import chain
import numpy as np
import pandas as pd
import os
from pandas import DataFrame
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, recall_score, precision_score
from sklearn.model_selection import cross_val_score

from Distributional import Word2VecFuncs, paramter_tuning
from Distributional.Word2VecFuncs import get_embedding
from sklearn.model_selection import train_test_split

# Folders

word2vec_model_folder = './Models/'
trial_path = "../Output/trial/"
sentence_folder = '../Dataset/sentences/'


def get_labeled_couples(path: str) -> DataFrame:
    couples = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split('\t')
        word1, word2, is_hyp = line[0].split('-')[0].strip(), line[1].split('-')[0].strip(), line[2]
        if is_hyp == 'True':
            couples.append([word1, word2, 1])
        else:
            couples.append([word1, word2, 0])
    return pd.DataFrame(couples, columns=['hypo', 'hyper', 'label'])


def split_train_test(path_to_labeled_dataset, w2v, test_size, random_state):
    labeled_couples = get_labeled_couples(path_to_labeled_dataset)
    labeled_couples['vec_a'] = labeled_couples['hypo'].apply(lambda x: get_embedding(w2v, '_'.join(x.split()))
    if len(x.split()) > 1 else get_embedding(w2v, x))
    labeled_couples['vec_b'] = labeled_couples['hyper'].apply(lambda x: get_embedding(w2v, '_'.join(x.split()))
    if len(x.split()) > 1 else get_embedding(w2v, x))
    embeddings = labeled_couples.dropna()
    X_train, X_test, y_train, y_test = train_test_split(embeddings.loc[:, ['hypo', 'hyper', 'vec_a', 'vec_b']],
                                                        embeddings.loc[:, ['label']], test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def evaluate_classifier(clf, threshold, X_test, y_test):
    predict = clf.predict_proba(self_concat(X_test))
    y_pred = [1 if x >= threshold else 0 for x in predict[:, 1]]
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("precision = %f, recall = %f" % (precision, recall))
    return precision, recall


def get_negative_embeddings(X_train, y_train, hhcouples: DataFrame):
    """
    Return none hypernym couples' embeddings which have hyponym or hypernym as the component of the couple
    :param X_train: Result of split_train_test
    :param y_train: Result of split_train_test
    :param hhcouples: extracted hypernym couples
    :return:
    """
    X_train = pd.concat([X_train, y_train], axis=1)
    X_train = X_train[X_train['label'] == 0]
    df1 = X_train[X_train['hypo'].isin(list(chain.from_iterable(hhcouples.values.tolist())))]
    df2 = X_train[X_train['hyper'].isin(list(chain.from_iterable(hhcouples.values.tolist())))]
    negative_embeddings = df1.append(df2)
    return negative_embeddings.loc[:, ['vec_a', 'vec_b']]


def merge_dataset(positive: DataFrame, negative: DataFrame) -> pd.DataFrame:
    """return a DataFrame that combines positive and negative hypernymy set
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
    combine = pd.DataFrame(np.column_stack(list(zip(*dataset['vec_a'])) + list(zip(*dataset['vec_b'])))).reset_index(
        drop=True)
    # lastcol = (dataset['vec_a'] - dataset['vec_b']) \
    #     .apply(lambda x: np.linalg.norm(x, ord=1)).reset_index(drop=True)
    #
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # result = pd.concat([combine, lastcol], axis=1)
    # result.columns = [i for i in range(0, 201)]
    # result[[200]] = result[[200]].apply(max_min_scaler)
    # return result
    return combine


def train_svm_model(dataset: DataFrame, show_cross_val=True):
    X_dataset = self_concat(dataset)
    y_dataset = dataset['label']
    clf = svm.SVC(probability=True)
    if show_cross_val:
        cross_val_score(clf, X_dataset, y_dataset, cv=5,
                              scoring=make_scorer(classification_report_with_precision_score))
    clf_res = clf.fit(X_dataset, y_dataset)
    return clf_res


def xgboost(dataset: DataFrame):
    X_dataset = self_concat(dataset)
    y_dataset = dataset['label']
    trial = paramter_tuning.create_study(X_dataset, y_dataset)
    dtrain = xgb.DMatrix(X_dataset, label=y_dataset)
    bst = xgb.train(trial.params, dtrain)
    return bst


def classification_report_with_precision_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return precision_score(y_true, y_pred)


def get_predict_result(nps: DataFrame, clf, isSaved=False, path_predict=None) -> DataFrame:
    """
    Predict with trained classifier and return a DataFrame of the predict couples and its probability being an HHcouple
    :param nps: predict set
    :param clf:
    :param isSaved:
    :param path_predict:
    :return: DataFrame, columns = NP_a, NP_b, y_prob_1
    """
    predict = clf.predict_proba(self_concat(nps))
    nps['y_prob_1'] = predict[:, 1].round(2)
    nps.sort_values('y_prob_1', inplace=True, ascending=False)
    nps.drop(columns=['vec_a', 'vec_b'], inplace=True)

    if isSaved:
        nps.to_csv(path_predict)
    return nps


def save_predict_set(nps: DataFrame, model, path):
    """ Save the NP pair embeddings in csv, to be run only once to build the massive matrix """
    print('>' * 12, 'Saving predict set ', '<' * 12)

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
    result.to_pickle("%s%d.pkl" % (path, count + 1))


def load_predict_set(path_to_predict_pair) -> DataFrame:
    print('>' * 12, 'Loading predict set ', '<' * 12)
    pairs = pd.concat([pd.read_pickle(path_to_predict_pair + x) for x in os.listdir(path_to_predict_pair)])
    return pairs
