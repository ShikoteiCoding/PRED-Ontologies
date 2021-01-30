import logging

from gensim.models import Word2Vec
from Distributional import Word2VecFuncs as w2vf
import Distributional.ML
import PatternFuncs as pf
import Distributional.W2VPhraser as w2vPhraser
import os
import pandas as pd

# Folders
dataset_path = "Dataset/"
processed_file_path = "Dataset/processed_files/"  # if run on whole corpus, pass this to the related generator to get line
spm_path = "Sequential Patterns/"
origin_sentences_path = 'Dataset/sentences/'
output_path = "Output/"
folder_of_new_corpus = "Output/sentences/"

w2v_path = "Output/word2vec/model/"
folder_of_parser = "Output/Phraser/"
iteration_path = "Output/Trial/iter_@/"  # replace @ with iteration folder

# File names
# tokenized file : 4298453 lines
test_corpus_name = "2B_music_bioreviews_tokenized_processed.txt"  # 473816 lines, 1/10 of the corpus for test purposes
spm_file_name = "sequential_patterns.txt"
couple_file_name = "Music.all"  # labeled couple dataset
HHCouple_file_name = "HHCouples.csv"
NPs_file_name = "NPs.txt"
original_sentences_file_name = 'sentence_00.txt'

# Total paths
path_to_test_corpus = dataset_path + test_corpus_name
path_to_whole_corpus = processed_file_path
path_to_original_sentences = origin_sentences_path + 'sliced_files/sentence_00.txt'  # For Phraser training
path_to_spm = spm_path + spm_file_name
path_to_HHCouple = iteration_path + HHCouple_file_name

# Thresholds
SP_TH = 0.75


def create_word2vec_model(path_to_sentence, model_name=None, isSave=True) -> Word2Vec:  # hyper parameter
    """
    return the word2vec model
    :param isSave: whether to save the model
    :param path_to_sentence: folder where the sentence file is stored (_tokenized.txt)
    :param model_name: specify the model name, if not it can be generated automatically after
    :return:
    """
    # setting hyper parameters
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context_size = 10  # context window size
    downsampling = 1e-3  # for frequent words

    if model_name is None:
        model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context_size)
    check_dir(w2v_path)
    path_to_model = w2v_path + model_name

    return w2vf.train_word2vec_model(path_to_sentence, path_to_model,
                                     num_workers, num_features, min_word_count, context_size, downsampling,
                                     isSave)


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
    iteration = 1
    path_to_iter = iteration_path.replace('@', str(iteration))
    check_dir(path_to_iter)

    # Step 0-1: Work with Phraser
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # enable logging
    check_dir(folder_of_parser)
    phrased_sentence_path = w2vPhraser.work_phraser(folder_of_parser, path_to_original_sentences, folder_of_new_corpus,
                                                    max_gram=4, min_counts=[50, 30, 20],
                                                    thresholds=[100, 1000, 1000])  # TODO to be tuned

    # Step 0-2: Train word2vec model
    # phrased_sentence_path = 'Dataset/sentences/4-grams-min20-threshold1000.txt'
    w2v = create_word2vec_model(phrased_sentence_path)
    print(pd.DataFrame(w2vf.get_topn_similar(w2v, "Lady_Gaga", 20)))
    print(pd.DataFrame(w2vf.get_topn_similar(w2v, "music", 10)))


    # Step 3: Extract HHCouples from hypernym_set
    list_of_patterns = pf.get_reliable_patterns(pf.parse_pattern_file(path_to_spm), SP_TH)
    HHCouples = pf.get_couples_from_patterns(path_to_test_corpus, list_of_patterns, hypernym_set,
                                             1000)  # path_to_whole_corpus for real use
    pf.save_extracted_couples(HHCouples, path_to_HHCouple)


    # Step 4: Get real NPs by filtering and crossing
    # w2v = w2vf.load_word2vec_model('Output/word2vec/300features_40minwords_10context')
    path_to_original_NPs = path_to_iter + NPs_file_name
    path_to_filtered_NPs = path_to_iter + NPs_file_name.replace('.txt', '-filtered.txt')
    pf.save_NPs(path_to_test_corpus, path_to_original_NPs)
    filtered = pf.filter_NP(path_to_original_NPs, path_to_filtered_NPs, 4, w2v)
    # TODO: Add:  filter with the [phraserName]-phrases.csv -> all the phrases calculated by phraser
    # print(len(filtered))
    """
    Step 5: Build dataset
      a. Get HHCouples vectors <-- HHCouple from file; w2vf.get_embedding(model, concept)
      b. Get negative set vectors <-- ML.get_negative_set(path); w2vf.get_embedding(model, concept)
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
