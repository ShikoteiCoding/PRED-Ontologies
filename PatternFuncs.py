import os
from typing import List, Generator
import gc
from typing import List, Tuple, Set

from gensim.models import Word2Vec

from Helpers import core_functions as cf
from Helpers import SP_matching as spm
from Helpers.Generator import LineGenerator, stop, BasicGenerator
from Helpers.HyperHypoCouple import HHCouple, NHHCouple
import pandas as pd


def get_couples_from_patterns(path_to_corpus: str, list_of_patterns: List[str], hypernym_set: List[str], limit: int) \
        -> (List[str]):
    """
    Extract a list of hypernymy couples based on high precision patterns and core concepts
    :param path_to_corpus: path of the corpus file
    :param list_of_patterns: patterns with precision above threshold
    :param hypernym_set: core concept as hypernym for the first iteration
    :param limit: max of sentences; for test proposes
    :return: list of hypernymy couples where the hypernym is in core_concepts
    """
    # Count
    count = 0

    # Store the couples to return in a list
    extracted_couples = []
    # Loop over the sentences
    sentences = cf.get_sentences_from_dir(path_to_corpus) if os.path.isdir(path_to_corpus) \
        else cf.get_sentences(path_to_corpus)

    for sentence in sentences:
        if count % 1000 == 0:
            print(count)
        if len(str(sentence)) > 500:
            continue
        # Limit to run test
        if count > limit:
            return extracted_couples

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

    print("count = ", count)
    return extracted_couples


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


def save_NPs(path_to_corpus: str, path_to_NPs: str):
    """ save NPs into a file; can be appended """
    NPs = set()
    count = 0
    # Loop over the sentences
    for sentence in cf.get_sentences_from_dir(path_to_corpus):
        if count % 5000 == 0:
            print("parsing %d sentences ..." % count)
        if len(str(sentence)) > 500:
            continue
        # add NPs of the sentence to set
        for np in sentence.NPs:
            NPs.add(cf.remove_first_occurrences_stopwords(np.text))
        count += 1
    with open(path_to_NPs, 'w', encoding='utf-8', errors='ignore') as f:
        for NP in iter(NPs):
            f.write(NP+'\n')


def filter_NP(path_to_input_NPs, output_path, max_length, model: Word2Vec):
    """    Filter the NPs with rules and those that are not in the vocabulary of the model    """
    saved_NPs = []
    with open(path_to_input_NPs, "r", encoding='utf-8', errors='ignore') as f:
        origin_NPs = f.readlines()
    for NP in origin_NPs:
        accept = False
        if '.' in NP:  # "compilation.The cons"
            NP = NP[:NP.index('.')]  # "compilation"
        if NP.isupper():  # MY LOINS TREMBLE
            continue
        if not NP.istitle():
            NP = NP.lower()
        words = NP.split()
        if len(words) > max_length:
            continue
        for word in words:
            if word.isdigit() or word in stop:
                accept = False
                break
        if NP or '_'.join(words) in model.wv:  # use a w2v model that is trained with a phraser-parsed corpus
            accept = True
        if accept:
            saved_NPs.append(NP)
    saved_NPs = set(saved_NPs)
    with open(output_path, 'w', encoding='utf-8', errors='ignore') as f_out:
        for NP in iter(saved_NPs):
            f_out.write(NP)
    return saved_NPs


def get_NPs_list(path_to_filteredNPs) -> List[str]:
    """ Return the filtered and unique NPs """
    with open(path_to_filteredNPs, 'r') as f:
        NPs = f.readlines()
    return list(set(NPs))


def save_extracted_couples(positive_set: List[str], path):
    df = pd.DataFrame(positive_set)
    df.drop_duplicates(inplace=True)
    df[2] = 'True'
    df.columns = ['hypo', 'hyper', 'label']
    df.to_csv(path, encoding='utf-8', index=False)


if __name__ == "__main__":
    list=[['a', 'b'], ['c', 'd']]
    df = pd.DataFrame(list)
    print(df)
    df[2]='True'
    df.columns=['hypo', 'hyper', 'label']
    print(df)
    df.to_csv('Output/test.csv')
