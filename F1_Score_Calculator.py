import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_cleaned.csv')
data.dropna(axis=0, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(data.loc[:, 'comment_text_clean'],
                                                    data.iloc[:, 2:8],
                                                    test_size=0.2,
                                                    random_state=43)

lda_train = pd.read_csv('lda_train_probs.csv', index_col=0)
lda_test = pd.read_csv('lda_test_probs.csv', index_col=0)
svm_train = pd.read_csv('svm_train_probs.csv', index_col=0)
svm_test = pd.read_csv('svm_test_probs.csv', index_col=0)

f1_lda = []
cutoff_lda = []
f1_svm = []
cutoff_svm = []


def find_cutoff(actual_matrix, prob_matrix, f1, cutoff):
    for i in range(6):
        best_f1 = 0
        best_cutoff = 0
        for j in np.arange(0, 1, 0.01):
            if f1_score(actual_matrix.iloc[:, i], prob_matrix.iloc[:, i] > j) > best_cutoff:
                best_f1 = f1_score(actual_matrix.iloc[:, i], prob_matrix.iloc[:, i] > j)
                best_cutoff = j
        f1.append(best_f1)
        cutoff.append(best_cutoff)


find_cutoff(y_train, lda_train, f1_lda, cutoff_lda)
find_cutoff(y_train, svm_train, f1_svm, cutoff_svm)

f1_lda_test = []
for i in range(6):
    score = f1_score(y_test.iloc[:, i], lda_test.iloc[:, i] > cutoff_lda[i])
    f1_lda_test.append(score)

f1_svm_test = []
for i in range(6):
    score = f1_score(y_test.iloc[:, i], svm_test.iloc[:, i] > cutoff_svm[i])
    f1_svm_test.append(score)


print(f1_lda_test)
print(f1_svm_test)



