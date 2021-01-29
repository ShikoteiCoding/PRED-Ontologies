import gc
from typing import List, Tuple, Set
from Helpers import core_functions as cf
from Helpers import SP_matching as spm
from Helpers.HyperHypoCouple import HHCouple, NHHCouple
import pandas as pd
import objgraph
import json

# Folders
dataset_path = "Dataset/"
spm_path = "Sequential Patterns/"

# File names
# tokenized file : 4298453 lines
corpus_file_name = "2B_music_bioreviews_tokenized_processed.txt"  # 473816 lines
first_corpus_file_name = "2B_music_bioreviews_tokenized_processed_part1.txt"
spm_file_name = "sequential_patterns.txt"
couple_file_name = "Music.all"  # labeled couple dataset
first_dataset_name = "first_dataset.csv"  # dataset of positive couples extracted from the first iteration + negative couples

# Total paths
path_to_corpus = dataset_path + corpus_file_name
path_to_first_corpus = dataset_path + first_corpus_file_name
path_to_spm = spm_path + spm_file_name
path_to_couple = dataset_path + couple_file_name
path_to_first_dataset = dataset_path + first_dataset_name
# Thresholds
SP_TH = 0.75


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
    f.close()
    return list_of_patterns


def select_patterns(patterns: List[Tuple[str, float]], SP_TH) -> List[str]:
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
    NPs = set()
    count = 0
    # Loop over the sentences
    for sentence in cf.get_sentences(path_to_corpus):
        if count % 5000 == 0:
            print(count)

        if len(str(sentence)) > 500:
            continue

        # add NPs of the sentence to set
        for np in sentence.NPs:
            NPs.add(cf.remove_first_occurrences_stopwords(np.text))
        count += 1
    print(len(NPs))
    with open(path_to_NPs, 'a', encoding='utf-8', errors='ignore') as f:
        json.dump(list(NPs), f)


def extract_patterns(path_to_corpus: str, list_of_patterns: List[str], core_concepts: List[str], limit: int) \
        -> (List[Tuple[HHCouple, str]], Set):
    """
    Extract a list of hypernymy couples based on high precision patterns and core concepts
    :param path_to_corpus: path of the corpus file
    :param list_of_patterns: patterns with precision above threshold
    :param core_concepts: core concept as hypernym
    :param limit: max of sentences; for test proposes
    :return: list of hypernymy couples and their pattern, where the hypernym is in core_concepts
    """
    # Count
    count = 0

    # Store the couples to return in a list
    extracted_couples = []
    NPs = set()
    # Loop over the sentences
    for sentence in cf.get_sentences(path_to_corpus):
        if count % 1000 == 0:
            print(count)

        if len(str(sentence)) > 500:
            continue

        # Limit to run test
        if count > limit:
            return extracted_couples, NPs

        sequence_representation = sentence.get_sequence_representation()

        # add NPs of the sentence to set
        for np in sentence.NPs:
            NPs.add(cf.remove_first_occurrences_stopwords(np.text))
        # Loop over the patterns
        for pattern in list_of_patterns:
            (done, hh_couples, pattern) = spm.spm_matching(pattern,
                                                           sequence_representation)  # :return bool, List[HHCouple], # pattern:
            if done:
                for hh_couple in hh_couples:
                    # if hh_couple.hypernym in list_of_patterns:
                    if hh_couple.hypernym in core_concepts:
                        extracted_couples.append((hh_couple, pattern))
                        # print(hh_couple)
            # del hh_couples
        # del sequence_representation
        count += 1

    print("count = ", count)
    return extracted_couples, NPs


def get_negative_set(path: str) -> List[NHHCouple]:
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
            negative_dataset.append(NHHCouple(word1, word2))
    f.close()
    return negative_dataset


def merge_dataset(positive_set: List[HHCouple], negative_set: List[NHHCouple]) -> pd.DataFrame:
    """
    return a DataFrame that combines positive and negative hypernymy set
    :param positive_set:
    :param negative_set:
    :return:
    """
    dataset_list = []
    for (HHCouple, pattern) in positive_set:
        dataset_list.append([HHCouple.hyponym, HHCouple.hypernym, 'True', pattern])
    for NHHCouple in negative_set:
        dataset_list.append([NHHCouple.nhypo, NHHCouple.nhyper, 'False', ""])
    df = pd.DataFrame(dataset_list)
    df.drop_duplicates(inplace=True)
    del dataset_list
    return df


def count_lines(file_name):
    """
    return the number of line of a file
    :param file_name:
    :return:
    """
    lines = 0
    for _ in open(file_name, 'rb'):
        lines += 1
    return lines


def printBat(fileNum):
    print("@echo off")
    for i in range(0, 10):
        s = 'origin_0' + str(fileNum) + '_0' + str(i) + '.txt'
        print('java -jar corpus_parsing.jar ', s)


def _save_NPs_in_folders(path_to_folder):
    for i in range(2, 10):
        corpus_path = path_to_folder + str(i) + "/" + "0%d_processed.txt" % i
        NP_path = path_to_folder + str(i) + "/" + "0%d_NPs.txt" % i

        save_NPs(corpus_path, NP_path)


if __name__ == "__main__":
    # _save_NPs_in_folders('Dataset/music_processed_0')
    save_NPs('Dataset/music_processed_09/09_processed.txt', 'Dataset/music_processed_09/09_NPs.txt')
    # # print(count_lines(path_to_corpus)) # 29110382
    # core_concepts = ["music"]
    # # tr = tracker.SummaryTracker()
    #
    # negative_set = get_negative_set(path_to_couple)
    # list_of_patterns = select_patterns(parse_pattern_file(path_to_spm), SP_TH)
    # # tr.print_diff()
    # print("----------------------------------------------------------------------------")
    # iter1_couples, NPs = extract_patterns(path_to_corpus, list_of_patterns, core_concepts, 1000)
    # # tr.print_diff()
    #
    # iter1_dataset = merge_dataset(iter1_couples, negative_set)
    # iter1_dataset.to_csv(path_to_first_dataset)
    # print(iter1_dataset[iter1_dataset[1] == 'music'])
    # # print(iter1_couples)
