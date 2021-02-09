import pandas as pd
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    positive_couples = pd.read_csv('Output/Trial test filter with no training set/iter_1/positive couples.csv').iloc[:, 1:4]
    extracted_couples = positive_couples[:2513].drop_duplicates()
    print(len(extracted_couples))
    predicted_couples = positive_couples[2514:]
    titles = [x for x in extracted_couples['hypo'] if x.istitle()]
    predicted_couples = pd.read_csv('Output/Trial test filter with no training set/iter_1/vector-predict.csv')
    genre = predicted_couples[predicted_couples['NP_b']=='genre']
    print(genre)
    TP = 7
    FP = len(predicted_couples[predicted_couples['NP_b']=='genre'])-TP
    # FN =
    precision = TP / (TP + FP)
    print(precision)


