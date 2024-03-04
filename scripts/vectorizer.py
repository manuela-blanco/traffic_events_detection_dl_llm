import csv
import os
import pickle
import time

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

truncate = 30

vector_models = [
    'Pt-BR_Word2Vec',
    'Pt-BR_GloVe',
    'Pt-BR_FastText',
]

pt_br_w2v = {
    50: 'word2vec_cbow_s50.txt',
    100: 'word2vec_cbow_s100.txt',
    300: 'word2vec_cbow_s300.txt',
    600: 'word2vec_cbow_s600.txt',
    1000: 'word2vec_cbow_s1000.txt',
}

pt_br_glove = {
    50: 'glove_s50.txt',
    100: 'glove_s100.txt',
    300: 'glove_s300.txt',
    600: 'glove_s600.txt',
    1000: 'glove_s1000.txt',
}

pt_br_fasttext = {
    50: 'fasttext_cbow_s50.txt',
    100: 'fasttext_cbow_s100.txt',
    300: 'fasttext_cbow_s300.txt',
    600: 'fasttext_cbow_s600.txt',
    1000: 'fasttext_cbow_s1000.txt',
}


class Word2VecModel():

    def __init__(
        self, train_dataset, test_dataset, class_qtd, base_model,
        embedding_path,
        set_dimensions=None,
    ):
        self.start_time = time.perf_counter()
        self.classdataset = class_qtd
        self.word2vec_type = base_model

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)

        if base_model == 'Pt-BR_Word2Vec':
            if set_dimensions:
                self.word2Vec_model = KeyedVectors.load_word2vec_format(
                    os.path.join(embedding_path, pt_br_w2v[set_dimensions]),
                    encoding='utf-8'
                )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Pt-BR_GloVe':
            if set_dimensions:
                self.word2Vec_model = KeyedVectors.load_word2vec_format(
                    os.path.join(embedding_path, pt_br_glove[set_dimensions]),
                    encoding='utf-8'
                )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Pt-BR_FastText':
            if set_dimensions:
                self.word2Vec_model = KeyedVectors.load_word2vec_format(
                    os.path.join(embedding_path, pt_br_fasttext[set_dimensions]),
                    encoding='utf-8'
                )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        print(f'The model has {self.dimension} dimensions')


    def parse_tweets(self, filename):
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            tweets = []
            for tweet in reader:
                tweets.append(tweet)
        return tweets


    def vectorize(self):
        train_labels = [int(tweet[0]) for tweet in self.train_tweets]
        test_labels = [int(tweet[0]) for tweet in self.test_tweets]

        if self.word2vec_type in vector_models:
            vectorizer = CountVectorizer(
                min_df=1,
                ngram_range=(1, 1),
                analyzer=u'word'
            )
            analyzer = vectorizer.build_analyzer()
            train_vectors = self.model_vectorize(
                tweet_base=self.train_tweets,
                analyzer=analyzer
            )
            test_vectors = self.model_vectorize(
                tweet_base=self.test_tweets,
                analyzer=analyzer
            )
        elif self.word2vec_type == 'random':
            train_vectors = self.random_vectorize(tweet_base=self.train_tweets)
            test_vectors = self.random_vectorize(tweet_base=self.test_tweets)

        print("{} word2vec matrix has been created as the input layer".format(
            self.word2vec_type
        ))

        vectorizing_time = time.perf_counter() - self.start_time
        return {
            'train_vectors': train_vectors,
            'train_labels': train_labels,
            'test_vectors': test_vectors,
            'test_labels': test_labels, 
            'vectorizing_time': vectorizing_time,
        }
    
    def prediction_only_vectorize(self, pred_tweets):
        pred_tweets = self.parse_tweets(pred_tweets)

        vectorizer = CountVectorizer(
            min_df=1,
            ngram_range=(1, 1),
            analyzer=u'word'
        )
        analyzer = vectorizer.build_analyzer()
        pred_vectors = self.model_vectorize(
            tweet_base=pred_tweets,
            analyzer=analyzer,
            tweet_index=0
        )

        print("{} word2vec matrix has been created as the input layer".format(
            self.word2vec_type
        ))

        return {
            'pred_vectors': pred_vectors,
        }

    def model_vectorize(self, tweet_base, analyzer, tweet_index=2):
        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )

        for i in range(len(tweet_base)):
            words_seq = analyzer(tweet_base[i][tweet_index])
            index = 0
            for word in words_seq:
                if index < self.tweet_length:
                    try:
                        values[i, index, :] = self.word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def random_vectorize(self, tweet_base):
        max_val = np.amax(self.word2Vec_model.syn0)
        min_val = np.amin(self.word2Vec_model.syn0)

        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )
        for i in range(len(tweet_base)):
            values[i, :, :] = min_val +\
                              (max_val - min_val) * np.random.rand(
                                                        self.tweet_length,
                                                        self.dimension
                                                    )
        return values

    def save(self, train_vectors, train_labels, test_vectors, test_labels, vectorizers_path, network_type):
        filename = vectorizers_path + self.word2vec_type + '_' + network_type + '_' + str(self.dimension) +\
                    '_' + self.classdataset + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [train_vectors, test_vectors, train_labels, test_labels],
                f
            )
