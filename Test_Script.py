import pandas as pd
import numpy as np
import pickle

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
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=500)
tfidf_vect.fit(x_train)
x_train_tfidf = tfidf_vect.transform(x_train)
x_test_tfidf = tfidf_vect.transform(x_test)

x_train_tfidf_os_all = []  # os = oversample
y_train_tfidf_os_all = []

# turn to array
x_train_tfidf = x_train_tfidf.toarray()
x_test_tfidf = x_test_tfidf.toarray()

for i in range(6):
    sm_tfidf = RandomOverSampler(random_state=40)
    x_train_tfidf_os, y_train_tfidf_os = sm_tfidf.fit_resample(x_train_tfidf, y_train.iloc[:, i])
    x_train_tfidf_os_all.append(x_train_tfidf_os)
    y_train_tfidf_os_all.append(y_train_tfidf_os)

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

    param_grid_SGD = {'penalty': ['l1', 'l2', 'elasticnet']}

    linear_SVM_obj = SGDClassifier(loss="modified_huber", max_iter=5)
    mod_linear_SVM = GridSearchCV(linear_SVM_obj,
                                  param_grid_SGD,
                                  scoring='f1',
                                  cv=5,
                                  refit=True,
                                  n_jobs=-1,
                                  verbose=0)

    mod_linear_SVM.fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

    print(mod_linear_SVM.best_params_)
    print(mod_linear_SVM.best_estimator_)

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
