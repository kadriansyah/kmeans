import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

class Clusterer:

    NEWLINE = '\n'
    WHITESPACE = ' '
    SKIP_FILES = {'cmds'}
    CORPUS_PATH  = 'corpus/articles/'
    K = range(80, 90)
    # NUM_CLUSTERS = 16
    NUM_CLUSTERS = 99

    def __init__(self):
        self.titles = []
        self.contents = []
        self.tfidf_matrix = None
        self.km = None
        self.frame = None
        self.scores = None

    '" read training files "'
    def __read_files(self, path):
        print('processing path: '+ path)
        for root, dir_names, file_names in os.walk(path):
            for dir_name in dir_names:
                self.__read_files(os.path.join(root, dir_name))
            for file_name in file_names:
                if file_name not in Clusterer.SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        past_title, titles, lines = False, [], []
                        f = open(file_path, encoding='latin-1')
                        for line in f:
                            if past_title == False:
                                titles.append(line)
                                past_title = True
                            elif past_title:
                                lines.append(line)
                        f.close()
                        title = Clusterer.NEWLINE.join(titles)
                        content = Clusterer.NEWLINE.join(lines)
                        yield file_path, title, content

    def __create_tfidf_matrix(self, content):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, use_idf=True, ngram_range=(1,3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(content)
        return tfidf_matrix

    def __prepare_data(self, path=CORPUS_PATH):
        for file_name, title, content in self.__read_files(path):
            print('processing title: '+ title)
            self.titles.append(title)
            self.contents.append(content)
        print('Creating TF-IDF Matrix...')
        self.tfidf_matrix = self.__create_tfidf_matrix(self.contents)
        joblib.dump(self.titles, 'pickled/_titles.pkl')
        joblib.dump(self.contents, 'pickled/_contents.pkl')
        joblib.dump(self.tfidf_matrix, 'pickled/_tfidf_matrix.pkl')

    def train_to_find_K(self, load=True):
        if load:
            self.titles = joblib.load('pickled/_titles.pkl')
            self.contents = joblib.load('pickled/_contents.pkl')
            self.tfidf_matrix = joblib.load('pickled/_tfidf_matrix.pkl')
        else:
            self.__prepare_data()
        print('Clustering to Find K...')
        self.km = [KMeans(n_clusters=i) for i in Clusterer.K]
        cluster_labels = [self.km[i].fit_predict(self.tfidf_matrix) for i in range(len(self.km))]
        self.scores = []
        for i in range(1, len(self.km)):
            score = silhouette_score(self.tfidf_matrix, cluster_labels[i])
            print("For n_clusters =", i, "The average silhouette_score is :", score)
            self.scores.append(score)

    def train(self, load=True):
        if load:
            self.titles = joblib.load('pickled/_titles.pkl')
            self.contents = joblib.load('pickled/_contents.pkl')
            self.tfidf_matrix = joblib.load('pickled/_tfidf_matrix.pkl')
        else:
            self.__prepare_data()
        self.km = KMeans(n_clusters=Clusterer.NUM_CLUSTERS)
        print('Clustering...')
        self.km.fit(self.tfidf_matrix)
        clusters = self.km.labels_.tolist()
        data = { 'titles': self.titles, 'contents': self.contents, 'cluster': clusters }
        self.frame = pd.DataFrame(data, index=[clusters] , columns=['titles', 'cluster'])

    def log_cluster(self):
        for i in range(Clusterer.NUM_CLUSTERS):
            print("Cluster %d titles:" % i, end='')
            for title in self.frame.ix[i]['titles'].values.tolist():
                print(' %s,' % title, end='')
            print() #add whitespace
            print() #add whitespace
