
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
path_to_whole_corpus = 'Dataset/processed_files/'
path_to_whole_sentences = 'Dataset/sentences/sliced_files/'
path_to_spm = spm_path + spm_file_name
path_to_w2v = w2v_path + w2v_model_name
path_to_np = np_path + np_file_name
path_to_np_lemmas = np_path + np_lemma_file_name
path_to_filtered_NPs = np_path + np_file_name.replace('.txt', '-above%d.txt' % MIN_NP_COUNT)
path_to_new_corpus = phraser_path + new_corpus_name
path_to_phraser = phraser_path + phraser_name
path_to_predict_set = np_path + predict_set_name.replace('.pkl', '/')
path_to_first_corpus_predict_set = np_path + 'pickles/first corpus, min=%d/' % MIN_NP_COUNT
