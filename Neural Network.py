import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM

toxic = pd.read_csv('train_cleaned.csv')
#toxic_test = pd.read_csv('test_cleaned.csv')

max_features = 2000

toxic.dropna(axis=0, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(toxic.loc[:,'comment_text_clean'], toxic.iloc[:,2:8], test_size = .3, random_state = 43)
#x_submission = toxic_test.loc[:,'comment_text_clean']
#x_submission = x_submission.fillna(' ')

#TF-IDF Vectors as features

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
tfidf_vect.fit(x_train)
x_train_tfidf = tfidf_vect.transform(x_train)
x_test_tfidf = tfidf_vect.transform(x_test)

x_train_tfidf_os_all = []
y_train_tfidf_os_all = []


for i in range(6):
    sm_tfidf = RandomOverSampler(random_state=40)
    x_train_tfidf_os, y_train_tfidf_os = sm_tfidf.fit_resample(x_train_tfidf, y_train.iloc[:,i])
    x_train_tfidf_os_all.append(x_train_tfidf_os)
    y_train_tfidf_os_all.append(y_train_tfidf_os)
#x_submission_tfidf = tfidf_vect.transform(x_submission)


data_dim = max_features
timesteps = 1
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

prediction_test = []
prediction_submission = []

x_test_tfidf = x_test_tfidf.toarray().reshape(x_test_tfidf.shape[0], 1, x_test_tfidf.shape[1])
#x_submission_tfidf = x_submission_tfidf.toarray().reshape(x_submission_tfidf.shape[0], 1, x_submission_tfidf.shape[1])

for i in range(6):
    x_train_tfidf = x_train_tfidf_os_all[i]
    x_train_tfidf = x_train_tfidf.toarray().reshape(x_train_tfidf.shape[0], 1, x_train_tfidf.shape[1])
    history = model.fit(x_train_tfidf, y_train_tfidf_os_all[i],
              batch_size=128, epochs=30,
              verbose=1,
              validation_split=0.1)
    model.save('my_model' + str(i) +'.h5')
    prediction_test.append(model.predict_proba(x_test_tfidf))
    #prediction_submission.append(model.predict_proba(x_submission_tfidf))

#prediction_submission_array = np.asarray(prediction_submission).reshape(6, 153164).transpose()

#submission = pd.DataFrame(data=prediction_submission_array,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'], index=toxic_test['id'])
#submission.to_csv('submission_1501.csv', index=True)

import ToxicModelBuilder
