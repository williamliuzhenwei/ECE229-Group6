import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os







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
df = df.drop_duplicates(subset=['book_title'])
#df = clean_df(df)
tfidf_vectors = get_tfidf_vectors(df)
print(recommendations(df.loc[df['book_title'] == 'Holes'].index[0], tfidf_vectors))
