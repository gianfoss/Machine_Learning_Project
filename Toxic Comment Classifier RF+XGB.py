# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RvPR3hm28qpliVs6qnKy_wGWT5TB3zW4
"""

#PARAMETERS
import os
import pandas as pd
import numpy as np
import sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#Utility stuff
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive, files
drive.mount('/content/gdrive')

np.random.seed(235)

#GLOBAL & CONSTANTS
GOOGLE_DRIVE_CODE_DIR = '/content/gdrive/My Drive/Code/'
SUBJECT_DIR = 'Machine Learning & Predictive Analytics/'
PROCESSED_DATA_DIR = 'data/Processed/'
MODEL_DIR = 'models/'
LOGS_DIR = 'logs/'

#NOTEBOOK SPECIFIC
HOMEWORK_DIR = 'Project/'
TRAIN_DATA = 'train_data_array.pkl'
TEST_DATA = 'test_data_array.pkl'
NOTEBOOK_NAME = 'Toxic Comment Classification.ipynb'


MAIN_PATH = os.path.join(GOOGLE_DRIVE_CODE_DIR
                    ,SUBJECT_DIR
                    ,HOMEWORK_DIR
                    )

INPUT_DATA_PATH = os.path.join(MAIN_PATH
                    ,PROCESSED_DATA_DIR
                    )

TEST_DATA = os.path.join(INPUT_DATA_PATH
                    ,TEST_DATA
                    )

TRAIN_PATH = os.path.join(INPUT_DATA_PATH
                    ,TRAIN_DATA
                    )

NOTEBOOK_FILE = os.path.join(MAIN_PATH
                    ,NOTEBOOK_NAME)

MODEL_EXPORT_PATH = os.path.join(MAIN_PATH
                    ,MODEL_DIR)

LOG_PATH = os.path.join(MAIN_PATH
                    ,LOGS_DIR)

import pickle
x_train_y_train_all_load = pickle.load(open(TRAIN_PATH, 'rb'))
x_test_y_test_all_load = pickle.load(open(TEST_DATA, 'rb'))

x_train_cv_os_all = x_train_y_train_all_load[0]
y_train_cv_os_all = x_train_y_train_all_load[1]

x_train_tfidf_os_all = x_train_y_train_all_load[2]
y_train_tfidf_os_all = x_train_y_train_all_load[3]

x_test_cv = x_test_y_test_all_load[0]
x_test_tfidf = x_test_y_test_all_load[2]
y_test = x_test_y_test_all_load[1]

y_test = [np.array(y_test.iloc[:,i]).reshape(-1,1) for i in range(6)]

class toxicmodel:
    def __init__(self, x_train, y_train, x_test, y_test, n = 6):
        self.n = n
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
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
            grid_search_model = GridSearchCV(model_name, hyper_param_grid, scoring = 'f1', cv = 5,refit = True, n_jobs=-1, verbose = 5)
            grid_search_model.fit(self.x_train[i], self.y_train[i])
            self.best_params.append(grid_search_model.best_params_)
            self.best_estimator.append(grid_search_model.best_estimator_)
    
    
    def predictmodel(self):
        for i in range(self.n):
            
            y_predict_train = self.best_estimator[i].predict(self.x_train[i])
            y_predict_test = self.best_estimator[i].predict(self.x_test)
             
            #y_predict_proba_train = self.best_estimator[i].predict_proba(self.x_train[i])[:,1]
            #y_predict_proba_test = self.best_estimator[i].predict_proba(self.x_test)[:,1]
            

            #self.y_predict_train.append(y_predict_train)
            #self.y_predict_test.append(y_predict_test)
            
            #self.y_predict_proba_train.append(y_predict_proba_train)
            #self.y_predict_proba_test.append(y_predict_proba_test)

            #self.roc_auc_score_train.append(roc_auc_score(self.y_train[i], y_predict_proba_train))
            #self.roc_auc_score_test.append(roc_auc_score(self.y_test[i], y_predict_proba_test))
            
            self.acc_score_train.append(accuracy_score(self.y_train[i], y_predict_train))
            self.acc_score_test.append(accuracy_score(self.y_test[i], y_predict_test))
            
            self.f1_score_train.append(f1_score(self.y_train[i], y_predict_train))
            self.f1_score_test.append(f1_score(self.y_test[i], y_predict_test))

            self.confusion_matrix_train.append(confusion_matrix(self.y_train[i], y_predict_train))
            self.confusion_matrix_test.append(confusion_matrix(self.y_test[i], y_predict_test))

            self.classification_report_train.append(classification_report(self.y_train[i], y_predict_train))
            self.classification_report_test.append(classification_report(self.y_test[i], y_predict_test))

n_estimators = []
max_features = []
max_depth = []

n_estimators.append(np.arange(100, 200, 50))
max_features.append(np.arange(1,5,1))
max_depth.append(np.array([1]))

n_estimators.append(np.arange(100, 200, 50))
max_features.append(np.arange(1,5,1))
max_depth.append(np.array([1]))

rf_param_grid = {'n_estimators':n_estimators[0]
                 ,'max_features':max_features[0]
                 ,'max_depth':max_depth[0]
                 ,'random_state':[235]
                }

xg_param_grid = {'objective':['binary:logistic']
                 ,'n_estimators':n_estimators[1]
                 ,'max_features':max_features[1]
                 ,'max_depth':max_depth[1]
                 ,'random_state':[235]
                #,'learning_rate': [0.15], #so called `eta` value
                #,'min_child_weight': [3,11],
                #'colsample_bytree': [0.5],
                }

if __name__ == '__main__':
    RF_toxic = toxicmodel(x_train_tfidf_os_all, y_train_tfidf_os_all, x_test_tfidf, y_test)
    RF_toxic.trainmodel(RandomForestClassifier(), rf_param_grid)
    RF_toxic.predictmodel()

    
    RF_toxic_pkl_path = os.path.join(MAIN_PATH
                                     ,MODEL_DIR
                                     ,'RF_toxic_pkl_path')
    RF_toxic_pkl = open(RF_toxic_pkl_path, 'wb')
    pickle.dump(RF_toxic, RF_toxic_pkl)
    
    XGB_toxic = toxicmodel(x_train_tfidf_os_all, y_train_tfidf_os_all, x_test_tfidf, y_test)
    XGB_toxic.trainmodel(XGBClassifier(), xg_param_grid)
    XGB_toxic.predictmodel()

    XGB_toxic_pkl_path = os.path.join(MAIN_PATH
                                      ,MODEL_DIR
                                      ,'XGB_toxic_pkl_path')
    
    XGB_toxic_pkl = open(XGB_toxic_pkl_path, 'wb')
    pickle.dump(XGB_toxic, XGB_toxic_pkl)