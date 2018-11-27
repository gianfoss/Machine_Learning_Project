import pandas as pd
import numpy as np
import pickle

from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def run_models(file, max_features=100, model='lda'):

    data = pd.read_csv(file)
    data.dropna(axis=0, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(data.loc[:, 'comment_text_clean'],
                                                        data.iloc[:, 2:8],
                                                        test_size=0.2,
                                                        random_state=43)

    # TF-IDF Vectors as features

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
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

    # model fit lists

    best_params = []
    best_estimator = []

    y_predict_train = []
    y_predict_test = []
    y_predict_proba_train = []
    y_predict_proba_test = []

    acc_score_train = []
    acc_score_test = []

    roc_auc_score_train = []
    roc_auc_score_test = []

    f1_score_train = []
    f1_score_test = []

    confusion_matrix_train = []
    confusion_matrix_test = []

    classification_report_train = []
    classification_report_test = []

    # fit models
    if model == 'lda':
        for i in range(6):
            # LDA Normal

            mod_lda_normal = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(x_train_tfidf_os_all[i],
                                                                                           y_train_tfidf_os_all[i])

            pred_train = mod_lda_normal.predict(x_train_tfidf)
            pred_test = mod_lda_normal.predict(x_test_tfidf)

            pred_proba_train = mod_lda_normal.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_lda_normal.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

    elif model == 'lda_shrink':
        for i in range(6):
            # LDA Shrinkage

            mod_lda_shrink = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(x_train_tfidf_os_all[i],
                                                                                             y_train_tfidf_os_all[i])

            pred_train = mod_lda_shrink.predict(x_train_tfidf)
            pred_test = mod_lda_shrink.predict(x_test_tfidf)

            pred_proba_train = mod_lda_shrink.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_lda_shrink.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

    elif model == 'qda':
        for i in range(6):
            # QDA

            mod_qda = QuadraticDiscriminantAnalysis().fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

            pred_train = mod_qda.predict(x_train_tfidf)
            pred_test = mod_qda.predict(x_test_tfidf)

            pred_proba_train = mod_qda.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_qda.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

    elif model == 'svm_linear':
        for i in range(6):
            # Linear SVM with grid search

            param_grid_linear = {'C': np.arange(0.01, 10.21, 0.2)}

            linear_SVM_obj = SVC(kernel='linear', probability=True)
            mod_linear_SVM = GridSearchCV(linear_SVM_obj,
                                          param_grid_linear,
                                          scoring='f1',
                                          cv=5,
                                          refit=True,
                                          n_jobs=-1,
                                          verbose=0)

            mod_linear_SVM.fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

            best_params.append(mod_linear_SVM.best_params_)
            best_estimator.append(mod_linear_SVM.best_estimator_)

            pred_train = mod_linear_SVM.best_estimator_.predict(x_train_tfidf)
            pred_test = mod_linear_SVM.best_estimator_.predict(x_test_tfidf)

            pred_proba_train = mod_linear_SVM.best_estimator_.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_linear_SVM.best_estimator_.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

    elif model == 'svm_poly':
        for i in range(6):
            # Polynomial SVM

            param_grid_poly = {'C': np.arange(0.01, 10.21, 0.2),
                               'degree': [2, 3, 4, 5, 6]}

            poly_SVM_obj = SVC(kernel='poly', probability=True)
            mod_poly_SVM = GridSearchCV(poly_SVM_obj,
                                        param_grid_poly,
                                        scoring='f1',
                                        cv=5,
                                        refit=True,
                                        n_jobs=-1,
                                        verbose=0)

            mod_poly_SVM.fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

            best_params.append(mod_poly_SVM.best_params_)
            best_estimator.append(mod_poly_SVM.best_estimator_)

            pred_train = mod_poly_SVM.best_estimator_.predict(x_train_tfidf)
            pred_test = mod_poly_SVM.best_estimator_.predict(x_test_tfidf)

            pred_proba_train = mod_poly_SVM.best_estimator_.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_poly_SVM.best_estimator_.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

    elif model == 'svm_radial':
        for i in range(6):
            # Radial Kernel SVM

            param_grid_rbf = {'C': np.arange(0.01, 10.21, 0.2),
                              'gamma': [0.1, 1, 10, 100]}

            rbf_SVM_obj = SVC(kernel='rbf', probability=True)
            mod_rbf_SVM = GridSearchCV(rbf_SVM_obj,
                                       param_grid_rbf,
                                       scoring='f1',
                                       cv=5,
                                       refit=True,
                                       n_jobs=-1,
                                       verbose=0)

            mod_rbf_SVM.fit(x_train_tfidf_os_all[i], y_train_tfidf_os_all[i])

            best_params.append(mod_rbf_SVM.best_params_)
            best_estimator.append(mod_rbf_SVM.best_estimator_)

            pred_train = mod_rbf_SVM.best_estimator_.predict(x_train_tfidf)
            pred_test = mod_rbf_SVM.best_estimator_.predict(x_test_tfidf)

            pred_proba_train = mod_rbf_SVM.best_estimator_.predict_proba(x_train_tfidf)[:, 1]
            pred_proba_test = mod_rbf_SVM.best_estimator_.predict_proba(x_test_tfidf)[:, 1]

            y_predict_train.append(pred_train)
            y_predict_test.append(pred_test)

            y_predict_proba_train.append(pred_proba_train)
            y_predict_proba_test.append(pred_proba_test)

            roc_auc_score_train.append(roc_auc_score(y_train.iloc[:, i], pred_proba_train))
            roc_auc_score_test.append(roc_auc_score(y_test.iloc[:, i], pred_proba_test))

            acc_score_train.append(accuracy_score(y_train.iloc[:, i], pred_train))
            acc_score_test.append(accuracy_score(y_test.iloc[:, i], pred_test))

            f1_score_train.append(f1_score(y_train.iloc[:, i], pred_train))
            f1_score_test.append(f1_score(y_test.iloc[:, i], pred_test))

            confusion_matrix_train.append(confusion_matrix(y_train.iloc[:, i], pred_train))
            confusion_matrix_test.append(confusion_matrix(y_test.iloc[:, i], pred_test))

            classification_report_train.append(classification_report(y_train.iloc[:, i], pred_train))
            classification_report_test.append(classification_report(y_test.iloc[:, i], pred_test))

        # Output arrays in a list

        name = model + '_output_' + str(max_features)
        output = [y_predict_proba_test, y_predict_proba_train, f1_score_test, f1_score_train]

        output_pkl = open(name, 'wb')
        pickle.dump(output, output_pkl)
        output_pkl.close()
