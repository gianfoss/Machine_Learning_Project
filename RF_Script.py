import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV


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


# svm probs

rf_predict_proba_train = []
rf_predict_proba_test = []

for i in range(6):
    # Linear SVM with grid search

    param_grid = {'n_estimators': [500, 750, 1000],
                  'max_features': [2, 4, 6, 8],
                  'max_depth': [2, 4, 6, 8],
                  'min_samples_split': [3, 4, 5, 6, 7, 8],
                  'random_state': [0]}

    # fit the model
    rf_obj = RandomForestClassifier()
    mod_grid = GridSearchCV(rf_obj, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1, verbose=5)
    mod_grid.fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

    pred_train = mod_grid.predict(x_train_tfidf)
    pred_test = mod_grid.predict(x_test_tfidf)

    pred_proba_train = mod_grid.predict_proba(x_train_tfidf)[:, 1]
    pred_proba_test = mod_grid.predict_proba(x_test_tfidf)[:, 1]

    rf_predict_proba_train.append(pred_proba_train)
    rf_predict_proba_test.append(pred_proba_test)

    print(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
    print(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

    print(accuracy_score(y_train.iloc[:, i], pred_train))
    print(accuracy_score(y_test.iloc[:, i], pred_test))

    print(f1_score(y_train.iloc[:, i], pred_train))
    print(f1_score(y_test.iloc[:, i], pred_test))

rf_test = np.asarray(rf_predict_proba_test).reshape(6, len(x_test)).transpose()
rf_train = np.asarray(rf_predict_proba_train).reshape(6, len(x_train)).transpose()

rf_test = pd.DataFrame(data=rf_test)
rf_train = pd.DataFrame(data=rf_train)

rf_test.to_csv('rf_test_probs.csv')
rf_train.to_csv('rf_train_probs.csv')
