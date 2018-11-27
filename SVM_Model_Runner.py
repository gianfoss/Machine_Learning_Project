import ClassificationModeler

ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='lda')

#ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='lda_shrink')

#ClassificationModeler.run_models('data/train_cleaned.csv', max_features=100, model='qda')
