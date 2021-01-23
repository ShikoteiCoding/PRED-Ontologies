from typing import List
from Helpers import core_functions as cf
from Helpers import SP_matching as spm

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

if __name__ == "__main__":
    list_of_patterns = parse_pattern_file(path_to_spm)

    for sentence in cf.get_sentences(path_to_corpus):
        print(sentence)

    #spm_matching(pattern, parsed_sentence, mingap = 0, maxgap = 3)



