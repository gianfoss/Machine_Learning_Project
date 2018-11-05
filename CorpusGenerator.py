import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

class CorpusGenerator:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"cant", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text

    def clean_data(self):
        self.data.drop(['id'], axis=1, inplace=True)

        # lowercase words and remove numbers
        self.data['comment_text'] = [re.sub('[^A-Za-z]', ' ', i).lower() for i in self.data['comment_text']]
        # remove contractions
        self.data['comment_text'] = [self.clean_text(i) for i in self.data['comment_text']]
        # tokenize words
        self.data['comment_text_tokenize'] = [word_tokenize(i) for i in self.data['comment_text']]

        # Stemming
        stemmer = SnowballStemmer('english')
        sentence_placeholder = []
        for sentence in self.data.loc[:, 'comment_text_tokenize']:
            sentence_stemmed = [stemmer.stem(self.clean_text(word)) for word in sentence]
            sentence_placeholder.append(sentence_stemmed)
        self.data['comment_text_tokenize_stemmed'] = sentence_placeholder

        # remove stopwords
        sentence_placeholder = []
        for sentence in self.data.loc[:, 'comment_text_tokenize_stemmed']:
            sentence_clean = [word for word in sentence if word not in stopwords.words('english')]
            sentence_placeholder.append(sentence_clean)
        self.data['comment_text_clean'] = sentence_placeholder
        self.data['comment_text_clean'] = [' '.join(i) for i in self.data['comment_text_clean']]

        # drop NA
        self.data.dropna(axis=0, inplace=True)
        return self.data

    def create_corpus(self, type = 'tf-idf', num_words = 1000):
        # Split train/test
        x_train, x_test, y_train, y_test = train_test_split(self.data.loc[:, 'comment_text_clean'], self.data.iloc[:, 1:7],
                                                            test_size=.3, random_state=43)

        if type == 'count_vectorizer':
            count_vect = CountVectorizer(max_features = num_words)
            count_vect.fit(x_train)
            x_train = count_vect.transform(x_train)
            x_test = count_vect.transform(x_test)
        elif type == 'tf-idf':
            tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features = num_words)
            tfidf_vect.fit(x_train)
            x_train = tfidf_vect.transform(x_train)
            x_test = tfidf_vect.transform(x_test)
            x_train = pd.DataFrame(x_train.todense())
            x_test = pd.DataFrame(x_test.todense())
        elif type == 'ngram':
            tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features = num_words)
            tfidf_vect_ngram.fit(x_train)
            x_train =  tfidf_vect_ngram.transform(x_train)
            x_test=  tfidf_vect_ngram.transform(x_test)
        else:
            print('Error: must input type as "count_vectorizor, tf-idf, or ngram')

        x_train_os_all = []
        y_train_os_all = []

        for i in range(6):
            ros = RandomOverSampler(random_state=40)
            x_train_os, y_train_os = ros.fit_resample(x_train, y_train.iloc[:,i])
            x_train_os_all.append(x_train_os)
            y_train_os_all.append(y_train_os)

        return x_train_os_all, y_train_os_all, x_test, y_test




