# coding: utf-8

# In[ ]:
import os
import pandas as pd
import numpy as np
import sklearn
import scipy.stats as sts
import time
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.utils import class_weight

import xgboost as xgb

#Utility stuff
#from google.colab import drive, files
#drive.mount('/content/gdrive')

#GLOBAL & CONSTANTS
ROOT_DIR = '/home/kniemi/'

#PATHS & NAMES
TRAIN_DATA = 'train_cleaned.csv'
PROCESSED_DATA_DIR = 'data/'
MODEL_DIR = 'models/'
LOGS_DIR = 'logs/'

INPUT_DATA_PATH = os.path.join(ROOT_DIR
                    ,PROCESSED_DATA_DIR
                    )

TRAIN_DATA = os.path.join(INPUT_DATA_PATH
                    ,TRAIN_DATA
                    )

MODEL_EXPORT_PATH = os.path.join(ROOT_DIR
                    ,MODEL_DIR)

LOG_PATH = os.path.join(ROOT_DIR
                    ,LOGS_DIR)


#GLOBAL MODEL PARAMETERS

max_features= 5000
grid_search_score_method = 'roc_auc'
stopping_rounds=100
#num_boost_round=999
cv = 4
iters = 20
tree='gpu_hist'

#MODEL TUNING PARAMETERS

max_depth = sts.randint(1,20)
#max_depth = 20
learn_rate = sts.uniform(0.02, 1.5)
n_estimators = sts.randint(100, 2000)
#n_estimators = 5000
min_child_weight = np.arange(1, 8, 1)
colsample_bytree = sts.uniform(0.4, 0.4)
subsample = sts.uniform(0.5, 0.3)
#min_split_loss = np.array([5])
reg_alpha = sts.randint(1, 100)
#reg_lambda = np.array([5000])
gamma = sts.uniform(0, 1.0)



param_space = {
             #  'objective' : 'binary:logistic'
                  'max_depth' : max_depth
               ,  'n_estimators': n_estimators
               ,  'learning_rate' : learn_rate
     #         ,  'min_child_weight' : min_child_weight
     #         ,  'subsample' : subsample
     #         ,  'colsample_bytree' : colsample_bytree
     #         ,  'random_state' : [235]
     #         ,  'gamma' : gamma
     #         ,  'reg_alpha': reg_alpha
             }
# In[ ]:
toxic = pd.read_csv(TRAIN_DATA)
toxic.dropna(axis=0, inplace=True)
X, x_test, y, y_test = train_test_split(toxic.loc[:,'comment_text_clean'], toxic.iloc[:,2:8], test_size = .2, random_state = 43)

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=235)

#TF-IDF Vectors as features

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
tfidf_vect.fit(x_train)
x_train_tfidf =  tfidf_vect.transform(x_train)
x_test_tfidf =  tfidf_vect.transform(x_test)
x_valid_tfidf = tfidf_vect.transform(x_valid)


x_train_tfidf_os_all = [] #os = oversample
y_train_tfidf_os_all = []


for i in range(6):
    sm_tfidf = RandomOverSampler(random_state=40)
    x_train_tfidf_os, y_train_tfidf_os = sm_tfidf.fit_resample(x_train_tfidf, y_train.iloc[:,i])
    x_train_tfidf_os_all.append(x_train_tfidf_os)
    y_train_tfidf_os_all.append(y_train_tfidf_os)


# In[ ]:
if __name__ == '__main__':
    
    #define function
    def train_random_xgb(X, y, space, fit):
         
        #build classifier
        #if no gpu available, change tree method to true
        
        clf = xgb.XGBClassifier(n_jobs=-1
                                , tree_method=tree
                                , random_state=42
                                , silent=True
                                , n_estimators=1000
                                #, eval_set=eval_set
                                #, eval_metric = 'auc'
                                )
        
        #create random search cross validation object, using parameter space, scoring, iters, cv, and scoring
        rs_clf = RandomizedSearchCV(clf
                                    , space
                                    , scoring=grid_search_score_method
                                    , n_iter = iters
                                    , verbose=1
                                    , random_state = 235
                                    , n_jobs=-1
                                    , refit=True
                                    , cv=cv
                                    )
        
        rs_clf.fit(X
                ,  y
                , verbose=True)

        return rs_clf

# In[ ]:

    #MODEL EVALUATION
    best_params = []
    best_model = []
    best_score = []
    best_f1_score = []
    cutoff_valid = []

    y_pred_proba_train = []
    y_pred_proba_test = []
    y_pred_proba_valid = []
    roc_score_test = []
    roc_score_valid = []
    f1_score_test = []
    f1_score_valid = []
    
    #FOR EACH CLASS RUN RANDOM GRID SEARCH EVALUATION
    
    for i in range(6):
        
        print('\n', 'Running Randomized Grid Search for Class:', i, '\n') 
        search_time_start = time.time()
        trained_model = train_random_xgb(x_train_tfidf_os_all[i]
                                        , y_train_tfidf_os_all[i]
                                        , param_space
                                        , i
                                        )
        print('\n','Grid Search Train Successful!')
        print('\n', 'RandomizedGridSearch minutes: ', (time.time() - search_time_start) / 60)

        #GET MODEL 
        best_params.append(trained_model.best_params_)
        best_model.append(trained_model.best_estimator_)
        best_score.append(trained_model.best_score_)
        
        y_pred_proba_train.append(trained_model.predict_proba(x_train_tfidf)[:,1])
        y_pred_proba_test.append(trained_model.predict_proba(x_test_tfidf)[:,1])
        y_pred_proba_valid.append(trained_model.predict_proba(x_valid_tfidf)[:,1])
        
        roc_score_valid.append(roc_auc_score(y_valid.iloc[:,i], y_pred_proba_valid[i]))
        roc_score_test.append(roc_auc_score(y_test.iloc[:,i], y_pred_proba_test[i]))       
        
        print('\n','Finding Best Cutoff & F1 Score')
        
        #CALCULATE BEST CUT OFF AND GET F1 SCORE FROM TRAIN
        #x_train, x_test, y_train, y_test = train_test_split(toxic.loc[:,'comment_text_clean'], toxic.iloc[:,2:8], test_size = .3, random_state = 43)
        
        search_time_start = time.time()
        for j in np.arange(0,1,0.001):
            cutoff = 0.0
            if f1_score(y_valid.iloc[:,i], (y_pred_proba_valid[i] > j)) > cutoff:
                f1 = f1_score(y_valid.iloc[:,i], (y_pred_proba_valid[i] > j))
        cutoff_valid.append(j)
        f1_score_valid.append(f1)   
       
        #f1_score_test.append(f1_score(y_test.iloc[:, i], (y_pred_proba_test[i] > 0.95)))
        f1_score_test.append(f1_score(y_test.iloc[:, i], (y_pred_proba_test[i] > cutoff_valid[i])))

        print('\n','Cutoff Calculation time:', (time.time()-search_time_start) / 60)

        #PRINT SUMMARY & SCORES 
        print('\n', 'Fit Summary:', i,'\n')
        print('\n', 'Best Params:', best_params[i]) 
        print('\n', 'Best Estimator:', best_model[i])
        print('\n', 'Best', grid_search_score_method, 'using CV of', cv, ':', best_score[i])
        print('\n', 'Validation Cutoff value:', cutoff_valid[i])
        print('\n', 'Validation ROC_AUC:', roc_score_valid[i])
        print('\n', 'Test ROC_AUC:', roc_score_test[i])
        print('\n', 'Validationn F1 Score:', f1_score_valid[i])
        print('\n', 'Test F1 score', f1_score_test[i])
        


    
