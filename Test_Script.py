from CorpusGenerator import CorpusGenerator
import pandas as pd
import numpy as np
import pickle

train = pd.read_csv('/home/ssgianfortoni/ML-project/data/train.csv')

corpus = CorpusGenerator(train)
corpus.clean_data()
x_train_list, y_train_list, x_test, y_test = corpus.create_corpus(num_words=1000)

# save outputs
x_train_1000_path = 'x_train_1000.pkl'
x_train_1000_pkl = open(x_train_1000_path, 'wb')
pickle.dump(x_train_list, x_train_1000_pkl)

y_train_1000_path = 'y_train_1000.pkl'
y_train_1000_pkl = open(y_train_1000_path, 'wb')
pickle.dump(y_train_list, y_train_1000_pkl)

x_test_1000_path = 'x_test_1000.pkl'
x_test_1000_pkl = open(x_test_1000_path, 'wb')
pickle.dump(x_test, x_test_1000_pkl)

y_test_1000_path = 'y_test_1000.pkl'
y_test_1000_pkl = open(y_test_1000_path, 'wb')
pickle.dump(y_test, y_test_1000_pkl)


# Close the pickle instances
x_train_1000_pkl.close()
y_train_1000_pkl.close()
x_test_1000_pkl.close()
y_test_1000_pkl.close()
