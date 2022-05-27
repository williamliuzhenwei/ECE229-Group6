import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import re
import string
import random
import pickle
import os


def clean_df(df):
    '''

    :param df:
    :return: removes non english books, fills missing Summaries and removes other weridness from Summary colm
    '''
    df = df.loc[df['Language'] == 'en']
    df.drop(['Unnamed: 0', 'user_id', 'location', 'age', 'isbn', 'rating', 'Language', 'Category', 'city', 'state',
             'country'], axis=1, inplace=True)
    df = df.drop_duplicates(subset=['book_title'])
    temp = df.loc[df['Summary'] == '9', 'Summary'] = df['book_title']
    df['Summary'] = df['Summary'].replace('\n', ' ', regex=True)
    df['Summary'] = df['Summary'].replace('&#39;', '\'', regex=True)
    df['Summary'] = df['Summary'].replace('&quot;', '', regex=True)
    df = df[df['Summary'].map(lambda x: x.isascii())]
    df = df[df['book_title'].map(lambda x: x.isascii())]
    df = df[df['book_author'].map(lambda x: x.isascii())]
    return df


def make_lower_case(text):
    '''

    :param text:
    :return: makes lower case
    '''
    assert isinstance(text, str)
    return text.lower()


def remove_stop_words(text):
    '''

    :param text:
    :return: removing stop words
    '''
    assert isinstance(text, str)
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def remove_punctuation(text):
    '''

    :param text:
    :return: removing punctuation
    '''
    assert isinstance(text, str)
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


def get_tfidf_vectors(df):
    '''

    :param df:
    :return: generated tfidf_vectors and saves off vector for future use
    '''
    tfidf_vectors = []
    if os.path.isfile("NLPwords.txt"):  # if file exists we have already generated a list just load
        with open("NLPwords.txt", 'rb') as f:
            tfidf_vectors = pickle.load(f)
        return tfidf_vectors
    else:
        title = df[['Summary']]
        title['Summary'] = title.Summary.apply(func=make_lower_case)
        title['Summary'] = title.Summary.apply(func=remove_stop_words)
        title['Summary'] = title.Summary.apply(func=remove_punctuation)

        corpus = []
        for words in title['Summary']:
            corpus.append(words.split())

        import gensim.downloader as api

        # Replace with your file loc
        # EMBEDDING_FILE = 'C:/Users/Gordon/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'
        # google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

        # Training our corpus with Google Pretrained Model

        # 100 to 400 based on runtime
        google_model = Word2Vec(vector_size=200, window=5, min_count=2, workers=-1)
        google_model.build_vocab(corpus)

        google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)
        # May what to increase n-grams
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=5, stop_words='english')
        tfidf.fit(title['Summary'])
        # Getting the words from the TF-IDF model
        tfidf_list = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
        tfidf_feature = tfidf.get_feature_names()
        line = 0
        # for each book description
        for desc in corpus:
            # Word vectors are of zero length
            sent_vec = np.zeros(200)
            # num of words with a valid vector in the book description
            weight_sum = 0
            # for each word in the book description
            for word in desc:
                if word in google_model.wv.key_to_index and word in tfidf_feature:
                    vec = google_model.wv[word]
                    tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))
                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
            if weight_sum != 0:
                sent_vec /= weight_sum
            tfidf_vectors.append(sent_vec)
            line += 1
        with open('NLPwords.txt', 'wb') as f:
            pickle.dump(tfidf_vectors, f)
        return tfidf_vectors


def recommendations(book_title, tfidf_vectors):
    '''

    :param book_title:
    :param tfidf_vectors:
    :param df:
    :return: index of most similar books on df
    '''
    n=5
    index = df.loc[df['book_title'] == book_title].index[0]
    # Perform cosine similarity book_title vs all other items in dataset
    cosine_similarities = cosine_similarity(tfidf_vectors, tfidf_vectors[index].reshape(1, -1))
    # sorting scores
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # grabbing top n
    sim_scores = sim_scores[1:(n+1)]
    book_indices = []
    for i in sim_scores:
        if i[1] > 0:
            book_indices.append(i[0])
    if len(book_indices) == 0:
        return None
    return book_indices


df = pd.read_csv('Preprocessed_data.csv')
df = clean_df(df)
tfidf_vectors = get_tfidf_vectors(df)
print(recommendations("Holes", tfidf_vectors))
