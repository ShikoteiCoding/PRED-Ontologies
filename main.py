import logging
from typing import Iterable

from gensim.models import Word2Vec
from pandas import DataFrame

from Distributional import Word2VecFuncs as w2vf
import Distributional.ML
import PatternFuncs as pf
import Distributional.W2VPhraser as w2vPhraser
import os
import pandas as pd


# Thresholds
SP_TH = 0.75
MIN_NP_COUNT = 10
MAX_NP_LENGTH = 4

""" phraser parameters """
max_gram = 4
min_counts = [5, 5, 5]
thresholds = [7, 6, 5]

""" word2vec parameters """
num_features = 100
min_word_count = 10
num_workers = 4
context_size = 10  # context window size
down_sampling = 1e-3  # for frequent words


# Folders
dataset_path = "Dataset/"
processed_file_path = "Dataset/processed_files/"  # if run on whole corpus, pass this to the related generator to get line
spm_path = "Sequential Patterns/"
origin_sentences_path = 'Dataset/sentences/'
np_path = "Output/NPs/"
w2v_path = "Output/word2vec/"
phraser_path = "Output/Phraser/"
iteration_path = "Output/Trial/iter_@/"  # replace @ with iteration folder


# File names
# test_corpus_name = "2B_music_bioreviews_tokenized_processed.txt"  # 473816 lines, 1/10 of the corpus for test purposes
spm_file_name = "sequential_patterns.txt"
corpus_file_name = "Music.all"  # labeled couple dataset
hhcouple_file_name = "HHCouples.csv"
np_file_name = "NPs.txt"
hypo_count_file_name = 'kept-hypos-count.csv'
hypo_in_vocab_name = 'hypo-in-vocab-result.csv'
w2v_model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context_size)
phraser_name = "%d-grams-min%d-threshold%d" % (max_gram, min_counts[-1], thresholds[-1])
new_corpus_name = phraser_name + '_sentences.txt'

# Total paths
# path_to_test_corpus = dataset_path + test_corpus_name
path_to_test_corpus = 'Dataset/processed_files/00_processed.txt'
path_to_whole_corpus = processed_file_path
path_to_original_sentences = origin_sentences_path + 'sliced_files/sentence_00.txt'  # For Phraser training
path_to_spm = spm_path + spm_file_name
path_to_w2v = w2v_path + w2v_model_name
path_to_np = np_path + np_file_name
path_to_kept_NPs = np_path + np_file_name.replace('.txt', '-above%d.txt' % MIN_NP_COUNT)
path_to_new_corpus = phraser_path + new_corpus_name
path_to_phraser = phraser_path + phraser_name


def evaluate_NP_vector(np_set, w2v: Word2Vec, isSaved=False, path_to_result=None) -> DataFrame:
    """
    Evaluate the representation of Phraser + w2v. Given NP set, see how many NP has a vector in word2vec
    :param np_set:
    :param w2v:
    :param path_to_result:
    :return:
    """
    matched = 0
    result = []
    for np in np_set:
        is_hypo_in_w2v = False
        if len(np.split()) > 1:
            np = '_'.join(np.split())
        if np in w2v.wv.vocab:
            is_hypo_in_w2v = True
            matched += 1
        result.append([np, is_hypo_in_w2v])
    print(result)
    df = pd.DataFrame(result, columns=['NP', 'is NP in word2vec'])
    if isSaved:
        df.to_csv(path_to_result)
    print("Matched NP: %d, NP set: %d, %f  " % (matched, len(np_set), matched / len(np_set)))
    return df


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print("create dir failed", str(e))


if __name__ == "__main__":
    # core_concepts = ["music"]
    core_concepts = ['music', 'activity', 'arrangement', 'instrument', 'composition', 'genre', 'instrumentation',
                     'artist', 'event', 'performance', 'singer', 'recording', 'group', 'record', 'label', 'remix']
    hypernym_set = core_concepts
    iteration = 2.2
    path_to_iter = 'Output/Trial/iter_@/'.replace('@', str(iteration))

    check_dir(path_to_iter)
    check_dir(np_path)
    check_dir(w2v_path)
    check_dir(phraser_path)

    # Paths for each iteration
    path_to_hhcouples = path_to_iter + hhcouple_file_name
    path_to_hypo_count = path_to_iter + hypo_count_file_name
    path_to_hypo_in_vocab = path_to_iter + hypo_in_vocab_name

    """     Step 0-1: Work with Phraser         """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # enable logging
    # phraser = w2vPhraser.work_phraser(phraser_path, path_to_original_sentences,
    #                                   max_gram=max_gram, min_counts=min_counts,
    #                                   thresholds=thresholds)
    phraser = w2vPhraser.load_phrases_model(path_to_phraser)
    """     Step 0-2: Train word2vec model      """
    # phraser = w2vPhraser.load_phrases_model(path_to_phraser)
    # print(path_to_new_corpus)
    # w2v = w2vf.train_word2vec_model(path_to_new_corpus, path_to_w2v,
    #                                 num_workers, num_features, min_word_count, context_size, down_sampling)
    w2v = w2vf.load_word2vec_model(path_to_w2v)
    #
    # """ Test with the w2v model"""
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "Lady_Gaga", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "electric_guitar", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "album", 20)))

    """     Step 3: Extract HHCouples from hypernym_set     """
    list_of_patterns = pf.get_reliable_patterns(pf.parse_pattern_file(path_to_spm), SP_TH)
    hhcouples = pf.get_couples_from_patterns(path_to_test_corpus, list_of_patterns, hypernym_set, 9999999, True, path_to_hhcouples)
    # hhcouples = pf.load_HHCouples_to_dataframe(path_to_hhcouples)


    # # Step 4: Get real NPs by filtering and crossing
    nps = pf.extract_NPs(path_to_test_corpus, MAX_NP_LENGTH, True, path_to_np)
    # list_nps = pf.load_NPs(path_to_NPs)

    # filter NPs with frequency
    kept_nps = pf.get_NPs_above_threshold(nps, MIN_NP_COUNT, True, path_to_kept_NPs)
    # the frequence of hypo as NP in the corpus
    hypo_counts = pf.get_hypos_with_count(hhcouples, kept_nps, True, path_to_hypo_count)
    # hypos = pd.read_csv(path_to_hypo_count)

    # whether the hypo is in word2vec's vocabulary
    evaluate_NP_vector(hypo_counts['hypo'].unique().tolist(), w2v, True, path_to_hypo_in_vocab)

    """
    Step 5: Build dataset
      a. Get HHCouples vectors <-- HHCouple from file; w2vf.get_embedding(model, concept)
    a  b. Get negative set vectors <-- ML.get_negative_set(path); w2vf.get_embedding(model, concept)
      c. Merge dataset <-- ML.merge_dataset(positive_set: List[List[float]], negative_set: List[List[float]]) -> pd.DataFrame
    
    Step 6: Train
      a. Boosting
      b. Split dataset
    
    Step 7: Construct predict dataset
      a. Get the intersaction of pattern-NPs and phrases-NPs
      b. Cartesian Product the string
      c. Get vector of the string    w2vf.get_embedding(model, concept) Note that for phrases it's "a_b" as parameters, not "a b"
      d. form the predict dataset
    
    Step 8: Predict
    need a new set to save the  predicted couples, to be removed in next training process
    Ste 9: Iterate
    
    """
