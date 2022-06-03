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

    #df = df[df['Summary'].map(lambda x: x.isascii())]
    #df = df[df['book_title'].map(lambda x: x.isascii())]
    #df = df[df['book_author'].map(lambda x: x.isascii())]
    #df.to_csv('file2.csv', header=True, index=True)
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
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_tfidf_vectors():
    '''

    :param df:
    :return: generated tfidf_vectors and saves off vector for future use
    '''
    tfidf_vectors = []
    if os.path.isfile("NLPwords.txt"):  # if file exists we have already generated a list just load
        #with open("NLPwords.json", 'r', encoding="utf-8") as f:
        #    tfidf_vectors = json.load(f)
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



        # Replace with your file loc
        # EMBEDDING_FILE = 'C:/Users/Gordon/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'
        # google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

        # Training our corpus with Google Pretrained Model

        # 100 to 400 based on runtime
        google_model = Word2Vec(vector_size=100, window=5, min_count=2, workers=-1)
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
        with open('NLPwords.txt', 'wb') as f:
            pickle.dump(tfidf_vectors, f)
        #with open('NLPwords.json', 'w', encoding="utf-8") as f:
        #    f.write(json.dumps(tfidf_vectors,cls=NumpyEncoder))

        return tfidf_vectors

#df must have droped duplicates
def recommendations(book_title_index, tfidf_vectors):
    '''

    :param book_title:
    :param tfidf_vectors:
    :param df:
    :return: index of most similar books on df must have droped duplicates
    !!!df = df.drop_duplicates(subset=['book_title'])!!!
    '''
    n=5
    #index = df.loc[df['book_title'] == book_title].index[0]
    index = book_title_index
    print(index)
    # Perform cosine similarity book_title vs all other items in dataset
    hold=tfidf_vectors[index]
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
print(df.nunique())
hold1=df.loc[df['book_title'] == 'Fahrenheit 451']
hold2=df.loc[df['book_title'] == 'Of Mice and Men']
hold3=df.loc[df['book_title'] == 'The Great Gatsby']
hold4=df.loc[df['book_title'] == 'The Grapes of Wrath']
tfidf_vectors = get_tfidf_vectors()

#temp = pd.DataFrame(tfidf_vectors)

#np.ndarray(shape=(),tfidf_vectors)
hold = recommendations(df.loc[df['book_title'] == 'Holes'].index[0], tfidf_vectors)


import csv
def get_book_title(book_indices):
    title=[]
    for i in book_indices:
        title.append(df.iloc[i] )
    return title

import csv
with open('index_to_title.csv','w',encoding='utf-8', newline="") as f:
    write = csv.writer(f)
    index = df.index
    i=0
    for book in df['book_title']:
        write.writerow([book, str(index[i])])
        i+=1


