import ClassificationModeler
import pickle

output = ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='lda')
name = '/home/ssgianfortoni/ML-project/lda_output_100.pkl'
output_pkl = open(name, 'wb')
pickle.dump(output, output_pkl)
output_pkl.close()

output = ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='lda_shrink')
name = '/home/ssgianfortoni/ML-project/lda_shrink_output_100.pkl'
output_pkl = open(name, 'wb')
pickle.dump(output, output_pkl)
output_pkl.close()

output = ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='qda')
name = '/home/ssgianfortoni/ML-project/qda_output_100.pkl'
output_pkl = open(name, 'wb')
pickle.dump(output, output_pkl)
output_pkl.close()
