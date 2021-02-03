import logging
from typing import Iterable, List

from gensim.models import Word2Vec
from pandas import DataFrame

from Distributional import Word2VecFuncs as w2vf, ML
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
num_features = 50
min_word_count = 10
num_workers = 4
context_size = 100  # context window size
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
predict_set_name = 'NPembeddings-predict-set.csv'
predict_result_name = 'vector-predict.csv'
# Total paths
# path_to_test_corpus = dataset_path + test_corpus_name
path_to_label_dataset = 'Dataset/' + corpus_file_name
path_to_test_corpus = 'Dataset/processed_files/00_processed.txt'
path_to_whole_corpus = processed_file_path
path_to_original_sentences = origin_sentences_path + 'sliced_files/sentence_00.txt'  # For Phraser training
path_to_spm = spm_path + spm_file_name
path_to_w2v = w2v_path + w2v_model_name
path_to_np = np_path + np_file_name
path_to_filtered_NPs = np_path + np_file_name.replace('.txt', '-above%d.txt' % MIN_NP_COUNT)
path_to_new_corpus = phraser_path + new_corpus_name
path_to_phraser = phraser_path + phraser_name
path_to_predict_set = np_path + predict_set_name


def evaluate_NP_vector(np_set:List, w2v: Word2Vec, isSaved=False, path_to_result=None) -> DataFrame:
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
    iteration = 2.4
    # 2.4 . reduce w2v feature size
    path_to_iter = 'Output/Trial/iter_@/'.replace('@', str(iteration))

    check_dir(path_to_iter)
    check_dir(np_path)
    check_dir(w2v_path)
    check_dir(phraser_path)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # enable logging

    # Paths for each iteration
    path_to_hhcouples = path_to_iter + hhcouple_file_name
    path_to_hypo_count = path_to_iter + hypo_count_file_name
    path_to_hypo_in_vocab = path_to_iter + hypo_in_vocab_name
    path_to_predict_result = path_to_iter + predict_result_name

    """     Step 0-1: Work with Phraser         """
    # phraser = w2vPhraser.work_phraser(phraser_path, path_to_original_sentences,
    #                                   max_gram=max_gram, min_counts=min_counts,
    #                                   thresholds=thresholds)
    # phraser = w2vPhraser.load_phrases_model(path_to_phraser)

    """     Step 0-2: Train word2vec model      """
    # w2v = w2vf.train_word2vec_model(path_to_new_corpus, path_to_w2v,
    #                                 num_workers, num_features, min_word_count, context_size, down_sampling)
    w2v = w2vf.load_word2vec_model(path_to_w2v)
    #
    # """ Test with the w2v model"""
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "guitar", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "electric_guitar", 30)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "jazz", 20)))

    """     Step 0-3: Get real NPs by filtering and crossing """
    # nps = pf.extract_NPs(path_to_test_corpus, MAX_NP_LENGTH, True, path_to_np)
    list_of_all_nps = pf.load_all_nps(path_to_np)  # All the NP words extracted by the parser

    # filter NPs with frequency
    dt_filtered_nps = pf.filter_nps(list_of_all_nps, MIN_NP_COUNT, isSave=True, path=path_to_filtered_NPs)
    # kept_nps = pf.load_filtered_nps(path_to_filtered_NPs)

    """     Step 3: Extract HHCouples from hypernym_set     """
    list_of_patterns = pf.get_reliable_patterns(pf.parse_pattern_file(path_to_spm), SP_TH)

    # dt_extracted_hhcouples_count = pf.get_couples_from_patterns(path_to_test_corpus, list_of_patterns, hypernym_set,
    #                                                             9999999, True, path_to_hhcouples)

    dt_extracted_hhcouples_count = pf.load_HHCouples_to_dataframe(path_to_hhcouples)

    # # # the frequency of hypo as NP in the corpus; works when keep_count = True in filter_nps()
    # hypos_in_filtered_np_set = pf.get_hypos_with_np_count(dt_extracted_hhcouples_count, dt_filtered_nps)
    # # # hypo_counts = pd.read_csv(path_to_hypo_count)
    # #
    # # # whether the hypo is in word2vec's vocabulary
    # evaluate_NP_vector(hypos_in_filtered_np_set, w2v, True, path_to_hypo_in_vocab)
    # dt_filtered_nps.drop('count')
    """     Step 5   Train the model     """
    print(">>>>>>>>>>>>>>>>>>>>>>>>> Train classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # Filter couples that hyponym is not in filtered NP set
    filtered_hhcouples = pf.get_filtered_hhcouples(dt_extracted_hhcouples_count, dt_filtered_nps)
    print(filtered_hhcouples)
    # TODO How to deal with the high quality (stats) couples ?
    filtered_hhcouples.drop(columns=['stats'], inplace=True)  # Do nothing and Drop for now
    hhcouple_embeddings = ML.return_features_from_word(filtered_hhcouples, w2v)
    # hhcouple_embeddings = ML.boost_embeddings(hhcouple_embeddings, 2)
    negative_embeddings = ML.get_embeddings_from_txt(path_to_label_dataset, w2v)

    train_dataset = ML.merge_dataset(hhcouple_embeddings, negative_embeddings)

    clf = ML.train_model(train_dataset, show_cross_val=False)

    del hhcouple_embeddings, negative_embeddings, train_dataset

    """     Step 6   Construct predict dataset """

    # print(kept_nps)
    print(">>>>>>>>>>>>>>>>>>>> building predict set >>>>>>>>>>>>>>>>>>>>>>>>>")
    # ML.save_predict_embedding_pairs(kept_nps, w2v, path_to_predict_set) # takes time, run once and for all

    predict_set = ML.build_predict_embedding_pairs(dt_filtered_nps, w2v, 100)
    # predict_set = ML.load_predict_embedding_pairs(path_to_predict_set)

    """     Step 7   Predict """
    print(">>>>>>>>>>>>>>>>>>>> Predict >>>>>>>>>>>>>>>>>>>>>>>>>")
    result = ML.get_predict_result(predict_set, clf, True, path_to_predict_result)
    pd.set_option('display.max_columns', None)

    print(result)
