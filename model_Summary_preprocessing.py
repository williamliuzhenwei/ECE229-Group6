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
import json
import csv
'''
Description
    Functions to preform all the preprocessing for our book recommandtion via book summary.
    Also saves the model and lookup table into two files index_to_title.csv & NLPwords.txt.
    NLPwords.txt contains the vectors of the model.
    index_to_title.csv is the lookup table.
    
    run clean_df - > get_tfidf_vectors -> index_to_title_generation
    user must provide the intial dataset
    
'''

def clean_df(df):
    '''
    Description
        removes non english books, fills missing Summaries with book title, removes duplicate books titles from df

    Args: df our base dataset
    Returns: df
    '''
    df = df.loc[df['Language'] == 'en']
    df.drop(['Unnamed: 0', 'user_id', 'location', 'age', 'isbn', 'rating', 'Language', 'Category', 'city', 'state',
             'country'], axis=1, inplace=True)
    df = df.drop_duplicates(subset=['book_title'])
    temp = df.loc[df['Summary'] == '9', 'Summary'] = df['book_title']
    df['Summary'] = df['Summary'].replace('\n', ' ', regex=True)
    df['Summary'] = df['Summary'].replace('&#39;', '\'', regex=True)
    df['Summary'] = df['Summary'].replace('&quot;', '', regex=True)
    return df


def make_lower_case(text: str):
    '''
    Description
        helper func to make text lower case

    Args: text
    Returns: text
    '''
    assert type(text) == str
    
    return text.lower()


def remove_stop_words(text: str):
    '''
    Description
        helper func to remove stop words (common english words that do not give information) for our NLP model

    Args: text
    Returns: text
    '''
    assert type(text) == str
    
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def remove_punctuation(text: str):
    '''
    Description
        helper func to remove punctuation from text

    Args: text
    Returns: text
    '''
    assert type(text) == str
    
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def get_tfidf_vectors(df):
    '''
    Description
        generates and returns tfidf_vectors. Saves off vector for future use in NLPwords.txt
        will load from file instead if it exists

    Args: df from cleaned_df
    Returns: 2D list of floats
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
        # Training our corpus with words to vec

        google_model = Word2Vec(vector_size=100, window=5, min_count=2, workers=-1)
        google_model.build_vocab(corpus)

        google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=5, stop_words='english')
        tfidf.fit(title['Summary'])
        # Getting the words from the TF-IDF model
        tfidf_list = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
        tfidf_feature = tfidf.get_feature_names()
        line = 0
        # for each book description
        for desc in corpus:
            # Word vectors are of zero length
            sent_vec = np.zeros(100)
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
        # dump tfidf_vectors into txt for easy loading
        with open('NLPwords.txt', 'wb') as f:
            pickle.dump(tfidf_vectors, f)
        return tfidf_vectors


def index_to_title_generation(df):
    '''
    Description
        saves a csv with row [title, index] use file along with tfidf_vectors to retrieve correct book title from index

    Args: df from cleaned_df
    Returns: None
    '''
    #index_to_title.csv = Preprocessed_data1.csv
    with open('index_to_title.csv','w',encoding='utf-8', newline="") as f:
        write = csv.writer(f)
        index = df.index
        i=0
        for book in df['book_title']:
            write.writerow([book, str(index[i])])
            i+=1
