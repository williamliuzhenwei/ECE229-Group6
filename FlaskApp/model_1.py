'''
Description
    Get the top N most similar books based on summary using cosine simmilarity.
    Finds similar books using NLPwords.txt and index_to_title.csv generated from model_Summary_preprocessing
    and returns their title and picture.
    This is the code that is runing on AWS so code is fast and light.
    Computation time for two functions under 1 secound
    
'''

from typing import Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



def recommendations(df: Any,
                    tfidf_vectors: list,
                    book_title: str):
    '''
    Description 
        Using cosine similarity finds N most similar books and returns their index as a list.
        Functions by geting the tfidf_vector at index of booktitle.
        Then we preform cosine similarity every tfidf_vector agaist our selected vector.
        Finally we pick the top N get the index and return it.
    Args:
        book_title: a book title must match exactly with book title in df
        tfidf_vectors: 2D list of floats that represent weight of each book vs each other from NLPwords.txt all values between 0 and 1
        df: Input a dataframe from index_to_title.csv
    
    Returns: 
        list of Index of N most similar books with the first item being the most similar on df or none if we cant find any books
    '''
    assert type(tfidf_vectors) == list
    assert type(book_title) == str
    assert len(book_title) == len(book_title.encode())
    assert book_title != ''
    
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
    Description 
        Retrieves book title and picture based on book index.
        Since the index returned from recommendations does not line up with main dataset
        we use our lookup table "index_to_title.csv" to convert the index for the main dataset.
        Then we can just select the row and get title and image.
    Args:
        book_indices: list of ints which represent the index of a book in our df
        df: Input a dataframe
    
    Returns: two lists. one that contains book title, another contains a link to a picture of the book
        
    '''
    assert type(book_indices) == list
    
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
    with open("NLPwords.txt", 'rb') as f:
        tfidf_vectors = pickle.load(f)
    # changed Preprocessed_data1.csv to index_to_title.csv
    df = pd.read_csv('index_to_title.csv')
    book_title = "Classical Mythology"
    book_indices = recommendations(df, tfidf_vectors, book_title)
    titles, imgs = get_book_title(df, book_indices)
