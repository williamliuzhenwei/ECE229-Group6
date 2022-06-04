'''
Description
    Finds similar books using NLPwords.txt and index_to_title.csv generated from model_Summary_preprocessing
    and returns their title
'''

from typing import Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



def recommendations(df: Any,
                    tfidf_vectors: list,
                    book_title: str):
    '''

    Description using cosine similarity finds N most similar books and returns their index as a list

    Args:
        book_title: a book title must match exactly with book title in df
        tfidf_vectors: 2D list of floats that represent weight of each book vs each other from NLPwords.txt
        df: Input a dataframe from index_to_title.csv
    
    Returns: 
        list of Index of N most similar books on df
    '''
    n = 5
    index = df.iloc[df.loc[df['book_title'] == book_title].index[0], 0]
    # Perform cosine similarity book_title vs all other items in dataset
    cosine_similarities = cosine_similarity(
        tfidf_vectors, tfidf_vectors[index].reshape(1, -1))
    # sorting scores
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # grabbing top n
    sim_scores = sim_scores[1:(n+1)]
    book_indices = [i[0] for i in sim_scores if i[1] > 0]
    if not book_indices:
        return None
    return book_indices


def get_book_title(df: Any,
                   book_indices: list):
    '''

    Description retrieves book title and picture based on book index


    Args:
        book_indices: list of ints which represent the index of a book in our df
        df: Input a dataframe
    
    Returns: two lists. one that contains book title, another contains a link to a picture of the book
        
    '''
    
    titles = []
    imgs = []
    for i in book_indices:
        title = df.iloc[i]['book_title']
        link = df.iloc[i]['img_s']
        titles.append(title)
        imgs.append(link)
    return titles, imgs


if __name__ == "__main __":
    import pickle
    import pandas as pd
    # changed NLPwords_final.txt to NLPwords.txt
    with open("NLPwords_final.txt", 'rb') as f:
        tfidf_vectors = pickle.load(f)
    # changed Preprocessed_data1.csv to index_to_title.csv
    df = pd.read_csv('index_to_title.csv')
    book_title = "Classical Mythology"
    book_indices = recommendations(df, tfidf_vectors, book_title)
    titles, imgs = get_book_title(df, book_indices)
