import json
import os
import re
from typing import List, Generator
import gc
from typing import List, Tuple, Set
import heapq

import spacy
from gensim.models import Word2Vec
from pandas import DataFrame

import Helpers
from Helpers import core_functions as cf
from Helpers import SP_matching as spm
from Helpers.Generator import LineGenerator, stop, BasicGenerator
from Helpers.HyperHypoCouple import HHCouple, NHHCouple
import pandas as pd


def get_couples_from_patterns(path_to_corpus: str, list_of_patterns: List[str], hypernym_set: Set, limit: int,
                              isSaved=False, path_to_hhcouples=None) \
        -> DataFrame:
    """
    Extract hypernymy couples based on high precision patterns and core concepts and calculate its times being extracted
    :param isSaved:
    :param path_to_corpus: path of the corpus file
    :param list_of_patterns: patterns with precision above threshold
    :param hypernym_set: core concept as hypernym for the first iteration
    :param limit: max of sentences; for test proposes
    :return: Dataframe of hypernymy couples, and the count of this couple being extracted
    """
    # Count
    count = 0

    # Store the couples to return in a list
    extracted_couples = []
    # Loop over the sentences
    sentences = cf.get_sentences_from_dir(path_to_corpus) if os.path.isdir(path_to_corpus) \
        else cf.get_sentences(path_to_corpus)

    for sentence in sentences:
        if count % 20000 == 0:
            print("Extracted %d couples from %d sentences " % (len(extracted_couples), count))
        if len(str(sentence)) > 500:
            continue


        sequence_representation = sentence.get_sequence_representation()

        # Loop over the patterns
        for pattern in list_of_patterns:
            (done, hh_couples, pattern) = spm.spm_matching(pattern,
                                                           sequence_representation)  # :return bool, List[HHCouple], # pattern:
            if done:
                for hh_couple in hh_couples:
                    if hh_couple.hypernym in hypernym_set:
                        extracted_couples.append([hh_couple.hyponym, hh_couple.hypernym])
        count += 1
        # Limit to run test
        if count > limit:
            break

    print("Parsed %d sentences " % count)
    df = pd.DataFrame(extracted_couples, columns=['hypo', 'hyper'], index=None)
    df['stats'] = df.groupby(['hypo', 'hyper'])['hypo'].transform('size')
    if isSaved:
        df.to_csv(path_to_hhcouples, encoding='utf-8', index=False)
    return df


def parse_pattern_file(path_to_corpus: str) -> List[Tuple[str, float]]:
    """
    Parse the sequential pattern file to return a list of patterns with precision
    :param path_to_corpus:
    :return List[str]:
    """
    list_of_patterns = []
    with open(path_to_corpus, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(";;")
        pattern = line[0]
        precision = float(line[1].replace('%', '')) / 100
        list_of_patterns.append((pattern, precision))
    return list_of_patterns


def get_reliable_patterns(patterns: List[Tuple[str, float]], SP_TH) -> List[str]:
    """
        Return a list of patterns with precision above the threshold
        :param patterns: List of patterns with precision; SP_TH: threshold
        :return List[str]:
    """
    list_of_patterns = []
    for line in patterns:
        pattern, precision = line[0], line[1]
        if precision > SP_TH:
            list_of_patterns.append(pattern)
    return list_of_patterns


def extract_NPs_lemma(list_of_all_nps: List[str]) ->List[str]:
    """
    Use spacy to get lemmas for a NP (word or phrase)
    :param list_of_all_nps: extracted NPs by java parser
    :return: list of NPs lemmatised by spacy
    """
    lemmas = []
    nlp = spacy.load("en_core_web_sm")
    for np in list_of_all_nps:
        np = nlp(np)
        for word in np.noun_chunks:
            lemmas.append(word.lemma_)

    return lemmas


def extract_NPs(path_to_corpus: str, max_length: int, isSave=False, path_to_NPs=None) ->List[str]:
    """
    Extract NPs from corpus, filter with max length, character and stop words
    :param path_to_corpus:
    :param max_length:
    :param isSave:
    :param path_to_NPs:
    :return:
    """
    print('>'*12, 'extract NPs ', '>'*12)

    NPs = []
    progress = 0
    # Loop over the sentences
    if os.path.isdir(path_to_corpus):
        g = cf.get_sentences_from_dir(path_to_corpus)
    else:
        g = cf.get_sentences(path_to_corpus)
    for sentence in g:
        if progress % 5000 == 0:
            print("Extracted %d NPs from %d sentences ..." % (len(NPs), progress))
        if len(str(sentence)) > 500:
            continue
        # Filter NPs
        for np in sentence.NPs:
            if len(np.text.split()) > max_length:
                continue
            if np.text in Helpers.Generator.stop:
                continue
            if re.search(r'[^a-zA-Z-\s]', np.text):  # accept only letters and - and ' in NP
                continue
            if np.text in Helpers.Generator.stop:
                continue

            NPs.append(cf.remove_first_occurrences_stopwords(np.text))
        progress += 1
    if isSave:
        with open(path_to_NPs, 'w', encoding='utf-8', errors='ignore') as f:
            for NP in NPs:
                f.write(NP+'\n')
    return NPs


def load_all_nps(path_to_NPs) -> List[str]:
    with open(path_to_NPs, 'r') as f:
        NPs = f.read().splitlines()
    return NPs


def filter_nps(np_list: List, min_count, keep_count=False, isSave=False, path=None) -> DataFrame:
    """
    get NPs that appear above n times in the corpus, and not in stop words
    :param keep_count:
    :param np_list:
    :param min_count: min appearance times
    :param isSave:
    :param path:
    :return: DataFrame['NP', 'count']
    """
    stop_words = set()
    for stop in Helpers.Generator.stop:
        stop_words.add(stop)
        stop_words.add(stop.capitalize())

    dt_count = pd.value_counts(np_list).rename_axis('NP').reset_index(name='count')
    dt_count = dt_count[dt_count['count'] >= min_count].drop_duplicates()
    dt_count['NP'] = dt_count['NP'].map(lambda x: x if x not in stop_words else None)
    dt_count.dropna(inplace=True)

    if not keep_count:
        dt_count.drop(columns=['count'], axis=1, inplace=True)
    if isSave:
        dt_count.to_csv(path, encoding='utf-8')
    return dt_count




def load_filtered_nps(path) -> DataFrame:
    return pd.read_csv(path, encoding='utf-8')


def load_HHCouples_to_dataframe(path) -> DataFrame:
    return pd.read_csv(path, encoding='utf-8')


def load_HHCouples_to_list(path):
    return load_HHCouples_to_dataframe(path).values.tolist()


def get_hypos_with_np_count(dt_couples: DataFrame, dt_NPs: DataFrame, isSave=False, path=None) -> List:
    """ For evaluation and result view use"""
    return pd.merge(dt_couples, dt_NPs, left_on='hypo', right_on='NP').sort_values('count', ascending=False)['hypo'].unique().tolist()



def get_filtered_hhcouples(dt_couples: DataFrame, dt_NPs: DataFrame, isSave=False, path=None) -> DataFrame:
    merge = pd.merge(dt_couples, dt_NPs, left_on='hypo', right_on='NP')
    if isSave:
        merge.loc[:, ['hypo', 'hyper', 'stats']].to_csv(path, encoding='utf-8')
    return merge.loc[:, ['hypo', 'hyper', 'stats']]

if __name__ == "__main__":

    # NPs = save_NPs('Dataset/processed_files/00_processed.txt', 'Dataset/NPs list/00_NPs_ori_list.txt')
    # top = pd.value_counts(NPs)
    # top.to_csv('Dataset/NPs list/00_NPs_count.csv')
    # print(top[:100])
    # filter_nps('Dataset/NPs list/00_NPs_count.csv',  30).to_csv('Dataset/NPs list/00_NPs_mincount30.csv')
    # NPs = pd.read_csv('Dataset/NPs list/00_NPs_count.csv')
    # NPs.columns=['np', 'count']
    # HHCouple = pd.read_csv('Output/trial/iter_1/HHCouples.csv')
    # HHCouple.columns=['np', 'hyper', 'label']
    # join = pd.merge(HHCouple, NPs).sort_values('count', ascending=False)
    # join.to_csv('Output/trial/iter_1/HHCouples count.csv')
    # print(join)
    # filter_nps('Dataset/NPs/NPs/01_NPs.txt', 100)
    df = pd.read_csv('Output/trial/iter_1/HHCouples count.csv')
    hypos = df[df['count']>10].np.unique()
    for x in df[df['count']>10].np.unique():
        print(x)
