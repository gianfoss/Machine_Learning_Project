import pandas as pd
import numpy as np
import pickle
import random

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = pd.read_csv('data/train_cleaned.csv')
data.dropna(axis=0, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(data.loc[:, 'comment_text_clean'],
                                                    data.iloc[:, 2:8],
                                                    test_size=0.2,
                                                    random_state=43)

# TF-IDF Vectors as features

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(x_train)
x_train_tfidf = tfidf_vect.transform(x_train)
x_test_tfidf = tfidf_vect.transform(x_test)

x_train_tfidf_os_all = []  # os = oversample
y_train_tfidf_os_all = []

for i in range(6):
    sm_tfidf = RandomOverSampler(random_state=40)
    x_train_tfidf_os, y_train_tfidf_os = sm_tfidf.fit_resample(x_train_tfidf, y_train.iloc[:, i])
    x_train_tfidf_os_all.append(x_train_tfidf_os)
    y_train_tfidf_os_all.append(y_train_tfidf_os)

# turn to array
for i in range(6):
    x_train_tfidf_os_all[i] = x_train_tfidf_os_all[i].toarray()

x_test_tfidf = x_test_tfidf.toarray()

# lda probs

#lda_predict_proba_train = []
#lda_predict_proba_test = []

# fit models
#for i in range(6):
#    # LDA Normal
#
#    mod_lda_normal = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(x_train_tfidf_os_all[i],
#                                                                                   y_train_tfidf_os_all[i])
#
#    pred_train = mod_lda_normal.predict(x_train_tfidf)
#    pred_test = mod_lda_normal.predict(x_test_tfidf)
#
#    pred_proba_train = mod_lda_normal.predict_proba(x_train_tfidf)[:, 1]
#    pred_proba_test = mod_lda_normal.predict_proba(x_test_tfidf)[:, 1]
#
#    lda_predict_proba_train.append(pred_proba_train)
#    lda_predict_proba_test.append(pred_proba_test)
#
#    print(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
#    print(roc_auc_score(y_test.iloc[:, i], pred_proba_test))
#
#    print(accuracy_score(y_train.iloc[:, i], pred_train))
#    print(accuracy_score(y_test.iloc[:, i], pred_test))
#
#    print(f1_score(y_train.iloc[:, i], pred_train))
#    print(f1_score(y_test.iloc[:, i], pred_test))
#
#lda_test = np.asarray(lda_predict_proba_test).reshape(6, len(x_test)).transpose()
#lda_train = np.asarray(lda_predict_proba_train).reshape(6, len(x_train)).transpose()
#
#lda_test = pd.DataFrame(data=lda_test)
#lda_train = pd.DataFrame(data=lda_train)
#
#lda_test.to_csv('lda_test_probs.csv')
#lda_train.to_csv('lda_train_probs.csv')

# svm probs

svm_predict_proba_train = []
svm_predict_proba_test = []

for i in range(6):
    # Linear SVM with grid search

    def batches(l, n):
        for j in range(0, len(l), n):
            yield l[j:j + n]

    # param_grid_SGD = {'penalty': ['l1', 'l2', 'elasticnet']}

    mod_linear_SVM = SGDClassifier(loss="modified_huber", penalty='l2')
    # mod_linear_SVM = GridSearchCV(linear_SVM_obj,
    #                              param_grid_SGD,
    #                              scoring='f1',
    #                              cv=5,
    #                              refit=True,
    #                              n_jobs=-1,
    #                              verbose=0)

    X = x_train_tfidf_os_all[i]
    Y = y_train_tfidf_os_all[i]

    shuffledRange = list(range(len(X)))
    n_iter = 5
    for n in range(n_iter):
        random.shuffle(shuffledRange)
        shuffledX = [X[j] for j in shuffledRange]
        shuffledY = [Y[j] for j in shuffledRange]
        for batch in batches(range(len(shuffledX)), 10000):
            mod_linear_SVM.partial_fit(shuffledX[batch[0]:batch[-1] + 1], shuffledY[batch[0]:batch[-1] + 1],
                                       classes=np.unique(Y))

    # print(mod_linear_SVM.best_params_)
    # print(mod_linear_SVM.best_estimator_)

    pred_train = mod_linear_SVM.predict(x_train_tfidf)
    pred_test = mod_linear_SVM.predict(x_test_tfidf)

    pred_proba_train = mod_linear_SVM.predict_proba(x_train_tfidf)[:, 1]
    pred_proba_test = mod_linear_SVM.predict_proba(x_test_tfidf)[:, 1]

    svm_predict_proba_train.append(pred_proba_train)
    svm_predict_proba_test.append(pred_proba_test)

    print(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
    print(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

    print(accuracy_score(y_train.iloc[:, i], pred_train))
    print(accuracy_score(y_test.iloc[:, i], pred_test))

    print(f1_score(y_train.iloc[:, i], pred_train))
    print(f1_score(y_test.iloc[:, i], pred_test))

svm_test = np.asarray(svm_predict_proba_test).reshape(6, len(x_test)).transpose()
svm_train = np.asarray(svm_predict_proba_train).reshape(6, len(x_train)).transpose()

svm_test = pd.DataFrame(data=svm_test)
svm_train = pd.DataFrame(data=svm_train)

svm_test.to_csv('svm_test_probs.csv')
svm_train.to_csv('svm_train_probs.csv')
