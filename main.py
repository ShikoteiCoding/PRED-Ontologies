from typing import List
from Helpers import core_functions as cf
from Helpers import SP_matching as spm
from Helpers.HyperHypoCouple import HHCouple
from Helpers import parsed_sentence as ps

# Folders
dataset_path = "Dataset/"
spm_path = "Sequential Patterns/"

# File names
corpus_file_name = "2B_music_bioreviews_tokenized_processed.txt"
spm_file_name = "sequential_patterns_Music.txt"

# Total paths
path_to_corpus = dataset_path + corpus_file_name
path_to_spm = spm_path + spm_file_name

def parse_pattern_file(path_to_corpus: str) -> List[str]:
    """
    Parse the sequential pattern file to return a list of patterns
    :param path_to_corpus:
    :return List[str]:
    """
    list_of_patterns = []
    with open(path_to_corpus, 'r') as f:
        lines = f.readlines()
    for line in lines:
        list_of_patterns.append(line.replace('\n', ''))
    return list_of_patterns

def extract_patterns(path_to_corpus: str, list_of_patterns: List[str], core_concepts: List[str], limit: int) -> List[HHCouple]:

    # Count
    count = 0

    # Store the couples to return in a list
    extracted_couples = []

    # Loop over the sentences
    for sentence in cf.get_sentences(path_to_corpus):

        # Limit to run test
        if count > limit:
            return extracted_couples

        # Loop over the patterns
        for pattern in list_of_patterns:
            (done, hh_couples, pattern) = spm.spm_matching(pattern, sentence) #  :return bool, List[HHCouple], pattern:

            if done:
                for hh_couple in hh_couples:
                    #if hh_couple.hypernym in list_of_patterns:
                    extracted_couples.append(hh_couple)

        count += 1
    return extracted_couples

if __name__ == "__main__":
    list_of_patterns = parse_pattern_file(path_to_spm)
    core_concepts = ["music"]
    iter1_couples = extract_patterns(path_to_corpus, list_of_patterns, core_concepts, 10)
    print(iter1_couples)


