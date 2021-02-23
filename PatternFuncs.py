import os
import re
from typing import List, Tuple, Set
import spacy
from gensim.models import Word2Vec
from pandas import DataFrame

import Helpers
from Helpers import core_functions as cf
from Helpers import SP_matching as spm
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
        if count % 500000 == 0:
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
        df.drop_duplicates().to_csv(path_to_hhcouples, encoding='utf-8', index=False)
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


def extract_NPs(path_to_corpus: str, max_length: int, path_to_NPs) -> List[str]:
    """
    Extract NPs from corpus, filter with max length, character and stop words
    :param path_to_corpus:
    :param max_length:
    :param isSave:
    :param path_to_NPs:
    :return:
    """
    print('>' * 12, 'extract NPs ', '<' * 12)

    NPs = []
    progress = 0
    # Loop over the sentences
    if os.path.isdir(path_to_corpus):
        g = cf.get_sentences_from_dir_NPlemma(path_to_corpus)
    else:
        g = cf.get_sentences_NPlemma(path_to_corpus)
    for sentence in g:
        if progress % 200000 == 0:
            print("Extracted %d NPs from %d sentences ..." % (len(NPs), progress))
        if len(str(sentence)) > 500:
            continue
        # Filter NPs
        for np in sentence.NPs:
            nps = cf.remove_first_occurrences_stopwords(np.text)
            if len(nps.split()) > max_length:
                continue

            if re.search(r'[^a-zA-Z-\s]', nps):  # accept only letters and - and ' in NP
                continue
            NPs.append(nps)
        progress += 1

    with open(path_to_NPs, 'w', encoding='utf-8', errors='ignore') as f:
        for NP in NPs:
            f.write(NP + '\n')
    return NPs


def load_all_nps(path_to_NPs) -> List[str]:
    with open(path_to_NPs, 'r') as f:
        NPs = f.read().splitlines()
    return NPs



def filter_nps(np_list: List, min_count) -> DataFrame:
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
    dt_count.drop(columns=['count'], axis=1, inplace=True)

    return dt_count


def get_filtered_hhcouples(dt_couples: DataFrame, dt_NPs: DataFrame, isSave=False, path=None) -> DataFrame:
    """
    Filter hypernym couples with filtered NP set to avoid having couples with unwanted NPs
    :param dt_couples:
    :param dt_NPs:
    :param isSave:
    :param path:
    :return:
    """
    merge = pd.merge(dt_couples, dt_NPs, left_on='hypo', right_on='NP')
    if isSave:
        merge.loc[:, ['hypo', 'hyper', 'stats']].to_csv(path, encoding='utf-8')
    return merge.loc[:, ['hypo', 'hyper', 'stats']]

