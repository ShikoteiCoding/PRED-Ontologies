import csv
import logging
from typing import Iterable, List, Dict

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
PREDICT_TH = 0.95  # Threshold whether to keep the predicted couples
EXIT_TH = 10  # Threshold whether to end iterations. The number of unique predicted hyponyms
MIN_NP_COUNT = 30
MAX_NP_LENGTH = 4

""" phraser parameters """
max_gram = 4
min_counts = [20, 20, 20]
thresholds = [0.2, 0.2, 0.2]

""" word2vec parameters """
num_features = 100
min_word_count = 20
num_workers = 4
context_size = 50  # context window size
down_sampling = 1e-3  # for frequent words

# Folders
dataset_path = "Dataset/"
processed_file_path = "Dataset/processed_files/"  # if run on whole corpus, pass this to the related generator to get line
spm_path = "Sequential Patterns/"
origin_sentences_path = 'Dataset/sentences/'
np_path = "Output/whole corpus/NPs/"
predict_npset_path = "Output/NPs/pickles/"
w2v_path = "Output/word2vec/whole corpus/"
phraser_path = "Output/Phraser/whole corpus/lemmatized/"
iteration_path = "Output/Trial/iter_@/"  # replace @ with iteration folder

# File names
# test_corpus_name = "2B_music_bioreviews_tokenized_processed.txt"  # 473816 lines, 1/10 of the corpus for test purposes
spm_file_name = "sequential_patterns.txt"
corpus_file_name = "Music.all"  # labeled couple dataset
hhcouple_file_name = "HHCouples.csv"
np_file_name = "All NPs-lemmatized-maxLength=%d.txt" % MAX_NP_LENGTH
np_lemma_file_name = np_file_name.replace('.txt', '-lemma.csv')
first_corpus_np_file_name = 'corpus01-NPs.txt'
hypo_count_file_name = 'kept-hypos-count.csv'
hypo_in_vocab_name = 'hypo-in-vocab-result.csv'
w2v_model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context_size)
phraser_name = "%d-grams-min%d-threshold%d" % (max_gram, min_counts[-1], thresholds[-1])
new_corpus_name = phraser_name + '_sentences.txt'
predict_set_name = 'pickles/NPembeddings-predict-set(whole corpus, min=%d ).pkl' % MIN_NP_COUNT
predict_result_name = 'vector-predict.csv'
positive_couples_name = 'positive couples.csv'
# Total paths
# path_to_test_corpus = dataset_path + test_corpus_name
path_to_label_dataset = 'Dataset/' + corpus_file_name
path_to_test_corpus = 'Dataset/processed_files/00_processed.txt'
path_to_input = 'Dataset/sentences/sentence_lemmatized.txt'
path_to_iteration_output = 'Output/Trial-Full NP-All hypos/iter_@/'
path_to_whole_corpus = 'Dataset/processed_files/'
path_to_whole_sentences = 'Dataset/sentences/sliced_files/'
path_to_spm = spm_path + spm_file_name
path_to_w2v = w2v_path + w2v_model_name
path_to_np = np_path + np_file_name
# path_to_np_lemmas = np_path + np_lemma_file_name
path_to_filtered_NPs = np_path + np_file_name.replace('.txt', '-above%d.txt' % MIN_NP_COUNT)
path_to_new_corpus = phraser_path + new_corpus_name
path_to_phraser = phraser_path + phraser_name
path_to_predict_set = predict_npset_path + np_file_name.replace('.txt', '/')
path_to_first_corpus_predict_set = np_path + 'pickles/first corpus, min=%d/' % MIN_NP_COUNT
path_to_phrased_corpus = phraser_path + "%d-grams-min%d-threshold%f_sentences.txt" % \
                     (max_gram, min_counts[-1], thresholds[-1])

def evaluate_NP_vector(np_set: List, w2v: Word2Vec, isSaved=False, path_to_result=None) -> DataFrame:
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


def check_dir(*paths):
    for path in paths:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                print("create dir failed", str(e))


def add_capital_s(list) -> List:
    result = []
    for element in list:
        result.append(element)
        result.append(element + 's')
        result.append(str(element + 's').title())
        result.append((str(element).title()))
    return result


def return_dict(list) -> Dict:
    dict = {}
    for word in list:
        dict[word] = word
        dict[word + 's'] = word
        dict[str(word + 's').title()] = word
        dict[str(word).title()] = word
    return dict


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    core_concepts = ['instrument', 'composition', 'genre', 'instrumentation', 'event']
    no_training_concepts = ['singer', 'artist', 'song', 'band', 'group', 'album', 'music', 'yes', 'people', 'musician',
                            'star', 'fan']

    hypernym_set = set(core_concepts)
    no_training_set = add_capital_s(no_training_concepts)
    iteration = 1
    check_dir(np_path,w2v_path, phraser_path, path_to_predict_set)

    # 2.4 . reduce w2v feature size

    """     Step 0-1: Work with Phraser         """
    if not os.path.exists(path_to_phrased_corpus):
        print("not exists path_to_phrased_corpus")
        path_to_phrased_corpus = w2vPhraser.work_phraser(phraser_path, path_to_input,
                                      max_gram=max_gram, min_counts=min_counts,
                                      thresholds=thresholds)

    """     Step 0-2: Train or load a word2vec model      """
    if not os.path.exists(path_to_w2v):
        print("not exists path_to_w2v")
        w2v = w2vf.train_word2vec_model(path_to_phrased_corpus, path_to_w2v,
                                        num_workers, num_features, min_word_count, context_size, down_sampling)
    else:
        w2v = w2vf.load_word2vec_model(path_to_w2v)

    # """ Test with the w2v model"""
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "rock", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "art_rock", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "electric_guitar", 20)))
    # print(pd.DataFrame(w2vf.get_topn_similar(w2v, "jazz", 20)))

    """     Step 0-3: Get real NPs by filtering and cross pairing """
    if not os.path.exists(path_to_np):
        print("not exists path_to_np")
        list_of_all_nps = pf.extract_NPs(processed_file_path, MAX_NP_LENGTH, path_to_np)
    else:
        list_of_all_nps = pf.load_all_nps(path_to_np)  # All the NP words extracted by the parser

    # filter NPs with frequency and stop words
    dt_filtered_nps = pf.filter_nps(list_of_all_nps, MIN_NP_COUNT)
    if not os.listdir(path_to_predict_set):
        print("not exists path_to_predict_set")
        ML.save_predict_set(dt_filtered_nps, w2v, path_to_predict_set)
    predict_set = ML.load_predict_set(path_to_predict_set)

    """     Step 3: Extract HHCouples from hypernym_set     """
    list_of_patterns = pf.get_reliable_patterns(pf.parse_pattern_file(path_to_spm), SP_TH)

    while True:
        path_to_iter = path_to_iteration_output.replace('@', str(iteration))
        check_dir(path_to_iter)
        # Paths for each iteration
        path_to_hhcouples = path_to_iter + hhcouple_file_name  # Extracted HHCouples
        path_to_predict_result = path_to_iter + predict_result_name  # Result of predict set
        path_to_positive_couples = path_to_iter + positive_couples_name  # positive couples extracted(stats = times being extracted) and predicted(stats = probability)
        positive_couple_set = pd.DataFrame(columns=['hypo', 'hyper', 'stats'])

        print('>'*12, " Extract HHCouples ", '<'*12)
        dt_extracted_hhcouples = pf.get_couples_from_patterns(path_to_whole_corpus, list_of_patterns,
                                                              return_dict(hypernym_set).keys(),
                                                              99999999, True, path_to_hhcouples)

        """     Step 4   Train the model     """
        # Filter couples that hyponym is not in filtered NP set
        X_train, X_test, y_train, y_test = ML.split_train_test(path_to_label_dataset, w2v, 0.25, 1)
        filtered_hhcouples = pf.get_filtered_hhcouples(dt_extracted_hhcouples, dt_filtered_nps).drop_duplicates()
        filtered_hhcouples['hyper'] = filtered_hhcouples['hyper'].apply(lambda x: return_dict(hypernym_set)[x]) # get the hypernym's lemma
        positive_couple_set = positive_couple_set.append(filtered_hhcouples)  # Add extracted couples to positive couple set

        print('>'*12, " Build training dataset ", '<'*12)
        filtered_hhcouples.drop(columns=['stats'], inplace=True)
        negative_embeddings = ML.get_negative_embeddings2(X_train, y_train, filtered_hhcouples)
        # negative_embeddings = ML.get_negative_embeddings(path_to_label_dataset, w2v,
        #                                                  filtered_hhcouples)  # Build negative dataset

        # Remove capital concepts and those that are in no_training set to avoid bias the classifier
        filtered_hhcouples = filtered_hhcouples.applymap(
            lambda x: None if x in no_training_set or str(x).istitle() else x)
        filtered_hhcouples.dropna(inplace=True)
        hhcouple_embeddings = ML.return_features_from_word(filtered_hhcouples, w2v)

        print("positive set size: %d " % hhcouple_embeddings.shape[0])
        print("negative set size: %d " % negative_embeddings.shape[0])

        train_dataset = ML.merge_dataset(hhcouple_embeddings, negative_embeddings)

        print('>'*12, " Train classifier ", '<'*12)
        clf = ML.train_svm_model(train_dataset, show_cross_val=True)  # SVM
        # clf = ML.xgboost(train_dataset, show_cross_val=True)

        """     Evaluate the prediction result      """
        ML.evaluate_classifier(clf, PREDICT_TH, X_test, y_test)

        del hhcouple_embeddings, negative_embeddings, train_dataset

        """     Step 6   Construct predict dataset """

        print('>'*12, " building predict set ", '<'*12)
        # Do inner join of all NP predict set and hypernym set to get the predict set of this iteration
        filtered_predict_set = pd.merge(predict_set, pd.DataFrame(hypernym_set, columns=['hyper']),
                                        left_on='NP_b', right_on='hyper', how='inner').drop(columns='hyper')

        """     Step 7   Predict     """
        print('>'*12, " Predict ", '<'*12)
        result = ML.get_predict_result(filtered_predict_set, clf, True, path_to_predict_result)

        """     Step 8 Save results of current iteration    """
        predicted_hhcouples = result[result['y_prob_1'] > PREDICT_TH]
        predicted_hhcouples.columns = ['hypo', 'hyper', 'stats']

        positive_couple_set = positive_couple_set.append(predicted_hhcouples)
        positive_couple_set.to_csv(path_to_positive_couples)

        """     Step 9  Start next iteration"""
        all_hypos = set([x for x in positive_couple_set['hypo'] if not str(x).istitle()])
        print("All hypos: ")
        print(all_hypos)
        new_discovered_hypos = all_hypos - (all_hypos & hypernym_set)
        print("new discovered hypos: ")
        print(new_discovered_hypos)
        if len(new_discovered_hypos) > EXIT_TH:
            for hypo in new_discovered_hypos:
                try:
                    if hypo not in hypernym_set:
                        hypernym_set.add(hypo)
                except:
                    continue
            iteration += 1
        else:
            break
