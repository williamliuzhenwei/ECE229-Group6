# -*- coding: utf-8 -*-
'''
This model contains different types of recommenders for our product
'''

# Commented out IPython magic to ensure Python compatibility.
from typing import Any
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

from nltk.corpus import stopwords
# nltk.download("stopwords")
# nltk.download('punkt')


def item_based_recommender(df: Any, book_title: str):
    '''This function is our item based recommender

    this model will look for books that have high correlations with the user's input according
    to book titles, and ratings from other users. Then it will generate the top five books
    that have the highest similarity.

    Args:
        df: Input a dataframe
        book_title: Input a book title

    Returns:
        5 book names with tile and pictures
    '''
    # test the book title is a string and contains only characters in the ascii table
    assert type(book_title) == str
    assert len(book_title) == len(book_title.encode())
    assert book_title != ''
    
    if book_title in df['book_title'].values:

        rating_counts = pd.DataFrame(df['book_title'].value_counts())
        rare_books = rating_counts[rating_counts['book_title'] <= 180].index
        common_books = df[~df['book_title'].isin(rare_books)]


        if book_title in rare_books:

            random = pd.Series(common_books['book_title'].unique()).sample(2).values
            print('There are no recommendations for this book')

        else:
            user_book_df = common_books.pivot_table(index=['user_id'],
                                                    columns=['book_title'],
                                                    values='rating')

            book = user_book_df[book_title]
            recom_data = pd.DataFrame(user_book_df.corrwith(book).sort_values(ascending=False)).reset_index(drop=False)

            if book_title in [book for book in recom_data['book_title']]:
                recom_data = recom_data.drop(recom_data[recom_data['book_title'] == book_title].index[0])

            low_rating = []
            for i in recom_data['book_title']:
                if df[df['book_title'] == i]['rating'].mean() < 5:
                    low_rating.append(i)

            if recom_data.shape[0] - len(low_rating) > 5:
                recom_data = recom_data[~recom_data['book_title'].isin(low_rating)]

            links = []
            titles = []
            recom_data = recom_data.head(5)
            for i in range(len(recom_data['book_title'].tolist())):
                url = df.loc[df['book_title'] == recom_data['book_title'].tolist()[i], 'img_l'][:1].values[0]
                title = df.loc[df['book_title'] == recom_data['book_title'].tolist()[i], 'book_title'][:1].values[0]
                links.append(url)
                titles.append(title)

            return titles, links

    else:
        print('Cant find book in dataset, please check spelling')


def content_based_recommender(df: Any, book_title: str):
    '''This is our content based recommender

    This system tries to generate the recommendation based on multiple features.
    Once the user input the book that he/she likes, we generate the feature vector
    including Title, Author, Publisher and genre. Then calculate the element wise
    squared distance between input vectors with other books’ feature vectors.
    Then the top 5 with the closest distance book will be recommended.

    Args:
        df: Input a dataframe
        book_title: Input a book title

    Returns:
        5 book names with tile and pictures

    '''
    # test the book title is a string and contains only characters in the ascii table
    assert type(book_title) == str
    assert len(book_title) == len(book_title.encode())
    assert book_title != ''

    if book_title in df['book_title'].values:
        rating_counts = pd.DataFrame(df['book_title'].value_counts())
        rare_books = rating_counts[rating_counts['book_title'] <= 100].index
        common_books = df[~df['book_title'].isin(rare_books)]

        if book_title in rare_books:

            random = pd.Series(
                common_books['book_title'].unique()).sample(2).values
            print('There are no recommendations for this book')
            print('Try: \n')
            print(f'{random[0]}', '\n')
            print(f'{random[1]}', '\n')

        else:

            common_books = common_books.drop_duplicates(subset=['book_title'])
            common_books.reset_index(inplace=True)
            common_books['index'] = list(range(common_books.shape[0]))
            target_cols = ['book_title',
                           'book_author', 'publisher', 'Category']
            common_books['combined_features'] = [' '.join(
                common_books[target_cols].iloc[i, ].values) for i in range(common_books[target_cols].shape[0])]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['combined_features'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['book_title']
                                 == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books = sorted(sim_books, key=lambda x: x[1],
                                      reverse=True)[1:6]

            books = [common_books[common_books['index'] == sorted_sim_books[i]
                                  [0]]['book_title'].item() for i in range(len(sorted_sim_books))]

            titles = []
            imgs = []
            for book in books:
                url = common_books.loc[common_books['book_title']
                                       == book, 'img_l'][:1].values[0]
                title = common_books.loc[common_books['book_title']
                                         == book, 'book_title'][:1].values[0]
                imgs.append(url)
                titles.append(title)
            return titles, imgs

    else:
        print('Cant find book in dataset, please check spelling')

#content_based_recommender('To Kill a Mockingbird')


def custom_recommender(df: Any, book_title: str):
    '''This is our item based recommender

    it contended not only item-based filtering using user’s rating and book title,
    content-based filtering using feature vector distance, and also it will be
    calculating the cosine similarly based on the book summary. The mixed model
    will first compare the features vectors including genre and author than compare
    the summary’s cosine similarity.

    Args:
        df: Input a dataframe
        book_title: Input a book title

    Returns:
        5 book names with tile and pictures

    '''
    #ITEM-BASED
    # test the book title is a string and contains only characters in the ascii table
    assert type(book_title) == str
    assert len(book_title) == len(book_title.encode())
    assert book_title != ''

    if book_title in df['book_title'].values:

        rating_counts = pd.DataFrame(df['book_title'].value_counts())
        rare_books = rating_counts[rating_counts['book_title'] <= 180].index
        common_books = df[~df['book_title'].isin(rare_books)]

        if book_title in rare_books:

            random = pd.Series(
                common_books['book_title'].unique()).sample(2).values
            print('There are no recommendations for this book')
            print('Try: \n')
            print(f'{random[0]}', '\n')
            print(f'{random[1]}', '\n')

        else:
            user_book_df = common_books.pivot_table(index=['user_id'],
                                                    columns=['book_title'], values='rating')

            book = user_book_df[book_title]
            recom_data = pd.DataFrame(user_book_df.corrwith(book).
                                      sort_values(ascending=False)).reset_index(drop=False)

            if book_title in list(recom_data['book_title']):
                recom_data = recom_data.drop(
                    recom_data[recom_data['book_title'] == book_title].index[0])

            low_rating = [i for i in recom_data['book_title']
                          if df[df['book_title'] == i]['rating'].mean() < 5]

            if recom_data.shape[0] - len(low_rating) > 5:
                recom_data = recom_data[~recom_data['book_title'].isin(
                    low_rating)]

            recom_data = recom_data[:1]
            recom_data.columns = ['book_title', 'corr']
            recommended_books = list(recom_data['book_title'])
            df_new = df[~df['book_title'].isin(recommended_books)]

            #CONTENT-BASED (Title, Author, Publisher, Category)
            rating_counts = pd.DataFrame(df_new['book_title'].value_counts())

            rare_books = rating_counts[rating_counts['book_title'] <= 100].index

            common_books = df_new[~df_new['book_title'].isin(rare_books)]
            common_books = common_books.drop_duplicates(subset=['book_title'])
            common_books.reset_index(inplace=True)
            common_books['index'] = list(range(common_books.shape[0]))
            target_cols = ['book_title',
                           'book_author', 'publisher', 'Category']
            common_books['combined_features'] = [' '.join(
                common_books[target_cols].iloc[i, ].values) for i in range(common_books[target_cols].shape[0])]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['combined_features'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['book_title']
                                 == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books = sorted(
                sim_books, key=lambda x: x[1], reverse=True)[1:2]

            books = [common_books[common_books['index'] == sorted_sim_books[i]
                                  [0]]['book_title'].item() for i in range(len(sorted_sim_books))]

            recommended_books.extend(iter(books))
            df_new = df_new[~df_new['book_title'].isin(recommended_books)]

            #CONTENT-BASED (SUMMARY)
            rating_counts = pd.DataFrame(df_new['book_title'].value_counts())
            rare_books = rating_counts[rating_counts['book_title'] <= 100].index
            common_books = df_new[~df_new['book_title'].isin(rare_books)]

            common_books = common_books.drop_duplicates(subset=['book_title'])
            common_books.reset_index(inplace=True)
            common_books['index'] = list(range(common_books.shape[0]))

            summary_filtered = []
            for i in common_books['Summary']:
                i = re.sub("[^a-zA-Z]", " ", i).lower()
                i = nltk.word_tokenize(i)
                i = [word for word in i if word not in set(
                    stopwords.words("english"))]

                i = " ".join(i)
                summary_filtered.append(i)

            common_books['Summary'] = summary_filtered
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['Summary'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['book_title']
                                 == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books2 = sorted(
                sim_books, key=lambda x: x[1], reverse=True)[1:4]
            sorted_sim_books = sorted_sim_books2[:2]
            summary_books = [common_books[common_books['index'] == sorted_sim_books[i]
                                          [0]]['book_title'].item() for i in range(len(sorted_sim_books))]

            recommended_books.extend(iter(summary_books))
            df_new = df_new[~df_new['book_title'].isin(recommended_books)]

            #TOP RATED OF CATEGORY
            category = common_books[common_books['book_title']
                                    == book_title]['Category'].values[0]
            top_rated = common_books[common_books['Category'] == category].groupby(
                'book_title').agg({'rating': 'mean'}).reset_index()

            if top_rated.shape[0] == 1:
                recommended_books.append(
                    common_books[common_books['index'] == sorted_sim_books2[2][0]]['book_title'].item())

            else:
                top_rated.drop(
                    top_rated[top_rated['book_title'] == book_title].index[0], inplace=True)
                top_rated = top_rated.sort_values(
                    'rating', ascending=False).iloc[:1]['book_title'].values[0]
                recommended_books.append(top_rated)

            imgs = []
            titles = []
            for recommended_book in recommended_books:

                url = df.loc[df['book_title'] ==
                             recommended_book, 'img_l'][:1].values[0]
                title = df.loc[df['book_title'] ==
                               recommended_book, 'book_title'][:1].values[0]

                imgs.append(url)
                titles.append(title)

            return titles, imgs
    else:
        print('Cant find book in dataset, please check spelling')


def main(mode=None):

    books = pd.read_csv('data/Preprocessed_data.csv')
    df = books.copy()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(columns=['Unnamed: 0', 'location', 'isbn',
                     'img_s', 'img_m', 'city', 'age',
                     'state', 'Language', 'country',
                     'year_of_publication'], axis=1, inplace=True)  # remove useless cols

    df.drop(index=df[df['Category'] == '9'].index, inplace=True)  # remove 9 in category

    df.drop(index=df[df['rating'] == 0].index, inplace=True)  # remove 0 in rating

    df['Category'] = df['Category'].apply(lambda x: re.sub('[\W_]+', ' ', x).strip())

    titles1, imgs1 = item_based_recommender(df,'Harry Potter and the Order of the Phoenix (Book 5)')
    print('item based recommender:', titles1, imgs1)
    titles2, imgs2 = content_based_recommender(df,'Harry Potter and the Order of the Phoenix (Book 5)')
    print('content based recommender', titles2, imgs2)
    titles3, imgs3 = custom_recommender(df,'Harry Potter and the Order of the Phoenix (Book 5)')
    print('mixed model recommender', titles3, imgs3)


if __name__ == "__main__":
    main()


