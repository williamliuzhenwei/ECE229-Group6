import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json


with open("NLPwords_final.txt", 'rb') as f:
    tfidf_vectors = pickle.load(f)
#search = pd.read_csv('index_to_title.csv')
#df.columns = ['book_title', 'index']
df = pd.read_csv('Preprocessed_data1.csv')
#df.to_csv('data_v3.csv')

def recommendations(book_title, tfidf_vectors):
    '''

    :param book_title:
    :param tfidf_vectors:
    :param df:
    :return: index of most similar books on df
    '''
    assert type(book_title) == str
    assert type(tfidf_vectors) == list
    
    n=5
    index = df.iloc[df.loc[df['book_title'] == book_title].index[0],0]
    print(index) 
    #index = row['index']
    #index = data[book_title]
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
    print(book_indices)
    return book_indices

def get_book_title(user_input):
    book_indices = recommendations(user_input, tfidf_vectors)
    titles = []
    imgs = []
    for i in book_indices:
        #df assuming no duplicate book titles and language = english
        title = df.iloc[i]['book_title']
        link = df.iloc[i]['img_s']
        titles.append(title)
        imgs.append(link)
    return titles, imgs

    
