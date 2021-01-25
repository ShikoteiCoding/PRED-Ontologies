from typing import List, Tuple
from Helpers import core_functions as cf
from Helpers import SP_matching as spm
from Helpers.HyperHypoCouple import HHCouple, NHHCouple
import pandas as pd
import gc
from Helpers import ParsedSentence as ps

# Folders
dataset_path = "Dataset/"
spm_path = "Sequential Patterns/"

# File names
corpus_file_name = "2B_music_bioreviews_tokenized_processed.txt"
spm_file_name = "sequential_patterns.txt"
couple_file_name = "Music.all"  # labeled couple dataset
first_dataset_name = "first_dataset.csv"

# Total paths
path_to_corpus = dataset_path + corpus_file_name
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
        list_of_patterns.append([pattern, precision])
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


def extract_patterns(path_to_corpus: str, list_of_patterns: List[str], core_concepts: List[str], limit: int) -> List[
    HHCouple]:
    # Count
    count = 0

    # Store the couples to return in a list
    extracted_couples = []

    # Loop over the sentences
    for sentence in cf.get_sentences(path_to_corpus):
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
                    # if hh_couple.hypernym in list_of_patterns:
                    if hh_couple.hypernym in core_concepts:
                        extracted_couples.append(hh_couple)
                        print(hh_couple)
            del hh_couples, pattern
        del sequence_representation
        count += 1
        if count % 100 == 0:
            gc.collect()
            if count % 500 == 0:
                print(count/100)
    print("count = ", count)
    return extracted_couples


def get_negative_set(path: str) -> List[NHHCouple]:
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


def merge_dataset(positive_set: List[HHCouple], negative_set: List[NHHCouple]) -> List[List[str]]:
    dataset_list = []
    for HHCouple in positive_set:
        dataset_list.append([HHCouple.hyponym, HHCouple.hypernym, 'True'])
    for NHHCouple in negative_set:
        dataset_list.append([NHHCouple.nhypo, NHHCouple.nhyper, 'False'])
    return dataset_list


def count_lines(file_name):
    lines = 0
    for _ in open(file_name, 'rb'):
        lines += 1
    return lines


if __name__ == "__main__":
    # print(count_lines(path_to_corpus)) # 29110382
    negative_set = get_negative_set(path_to_couple)
    list_of_patterns = select_patterns(parse_pattern_file(path_to_spm), SP_TH)

    core_concepts = ["music"]
    iter1_couples = extract_patterns(path_to_corpus, list_of_patterns, core_concepts, 99999999)
    merged_dataset = merge_dataset(iter1_couples, negative_set)
    df = pd.DataFrame(merged_dataset)
    df.to_csv(path_to_first_dataset)
    print(df[df[1] == 'music'])
    # print(iter1_couples)

