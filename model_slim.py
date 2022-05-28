import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
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
