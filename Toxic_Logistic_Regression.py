import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import class_weight

import re


toxic = pd.read_csv('train_cleaned.csv')

max_features = 1000

toxic.dropna(axis=0, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(toxic.loc[:,'comment_text_clean'], toxic.iloc[:,2:8], test_size = .2, random_state = 43)

#TF-IDF Vectors as features

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
tfidf_vect.fit(x_train)
x_train_tfidf =  tfidf_vect.transform(x_train)
x_test_tfidf =  tfidf_vect.transform(x_test)

x_train_tfidf_os_all = [] #os = oversample
y_train_tfidf_os_all = []


for i in range(6):
    sm_tfidf = RandomOverSampler(random_state=40)
    x_train_tfidf_os, y_train_tfidf_os = sm_tfidf.fit_resample(x_train_tfidf, y_train.iloc[:,i])
    x_train_tfidf_os_all.append(x_train_tfidf_os)
    y_train_tfidf_os_all.append(y_train_tfidf_os)

#########
#Start building model here

class toxicmodel:
    def __init__(self, x_train, y_train, x_test, y_test, x_train_tfidf, n=6):
        self.n = n
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_tfidf = x_train_tfidf

        self.best_params = []
        self.best_estimator = []

        self.y_predict_train = []
        self.y_predict_test = []
        self.y_predict_proba_train = []
        self.y_predict_proba_test = []

        self.acc_score_train = []
        self.acc_score_test = []

        self.roc_auc_score_train = []
        self.roc_auc_score_test = []

        self.f1_score_train = []
        self.f1_score_test = []

        self.confusion_matrix_train = []
        self.confusion_matrix_test = []

        self.classification_report_train = []
        self.classification_report_test = []


    def trainmodel(self, model_name, hyper_param_grid):
        for i in range(self.n):
            grid_search_model = RandomizedSearchCV(model_name, hyper_param_grid, scoring='roc_auc', cv=5, refit=True, n_jobs=-1, n_iter=10,
                                             verbose=5)
            grid_search_model.fit(self.x_train[i], self.y_train[i])
            self.best_params.append(grid_search_model.best_params_)
            self.best_estimator.append(grid_search_model.best_estimator_)


    def predictmodel(self):
        for i in range(self.n):
            y_predict_train = self.best_estimator[i].predict(self.x_train_tfidf)
            y_predict_test = self.best_estimator[i].predict(self.x_test)

            y_predict_proba_train = self.best_estimator[i].predict_proba(self.x_train_tfidf)[:, 1]
            y_predict_proba_test = self.best_estimator[i].predict_proba(self.x_test)[:, 1]

            self.y_predict_train.append(y_predict_train)
            self.y_predict_test.append(y_predict_test)

            self.y_predict_proba_train.append(y_predict_proba_train)
            self.y_predict_proba_test.append(y_predict_proba_test)

            #self.roc_auc_score_train.append(roc_auc_score(self.y_train[i], y_predict_proba_train))
            self.roc_auc_score_test.append(roc_auc_score(self.y_test.iloc[:,i], y_predict_proba_test))

            #self.acc_score_train.append(accuracy_score(self.y_train[i], y_predict_train))
            #self.acc_score_test.append(accuracy_score(self.y_test.iloc[:,i], y_predict_test))

            #self.f1_score_train.append(f1_score(self.y_train[i], y_predict_train))
            #self.f1_score_test.append(f1_score(self.y_test.iloc[:,i], y_predict_test))

            #self.confusion_matrix_train.append(confusion_matrix(self.y_train[i], y_predict_train))
            #self.confusion_matrix_test.append(confusion_matrix(self.y_test.iloc[:,i], y_predict_test))

            #self.classification_report_train.append(classification_report(self.y_train[i], y_predict_train))
            #self.classification_report_test.append(classification_report(self.y_test.iloc[:,i], y_predict_test))



rf = toxicmodel(x_train_tfidf_os_all, y_train_tfidf_os_all, x_test_tfidf, y_test, x_train_tfidf, n=6)
rf.trainmodel(LogisticRegression(), {'random_state':[0]
                                         })
rf.predictmodel()

train_f1 = []
cutoff = []
for i in range(6):
    best_f1 = 0
    best_cutoff = 0
    for j in np.arange(0,1,0.001):
        if f1_score(y_train.iloc[:,i], rf.y_predict_proba_train[i] > j) > best_cutoff:
            best_f1 = f1_score(y_train.iloc[:,i], rf.y_predict_proba_train[i] > j)
            best_cutoff = j
    train_f1.append(best_f1)
    cutoff.append(best_cutoff)

test_f1 = [f1_score(y_test.iloc[:,i], rf.y_predict_proba_test[i] > cutoff[i]) for i in range(6)]
print(train_f1)
print(test_f1)
print(rf.roc_auc_score_test)
print(rf.best_estimator)