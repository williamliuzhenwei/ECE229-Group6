import pandas as pd
import numpy as np
#df = pd.read_csv('final_dataset.csv')
#df.drop(['image_url','website','twitter'])

#df2 = pd.read_csv('books.csv')
df = pd.read_csv('good_reads_final.csv')
df['birthplace'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)
df['book_title'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)
df['author_name'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)
df['birthplace'] = df['birthplace'].str.strip()
df['book_title'] = df['book_title'].str.strip()
df['author_genres'] = df['author_genres'].str.replace(',+$', '')
df['publish_date'] = df['publish_date'].replace(r'^\s*$', np.nan, regex=True)
df['publish_date'] = df['publish_date'].str[-4:]
df['publish_date'] = df['publish_date'].replace('by', np.nan, regex=True)
df['birthplace']=df['birthplace'].replace(r'^\s*$', np.nan, regex=True)
df['publish_date'] = pd.to_numeric(df['publish_date'], errors='coerce')
df['book_id'] = df['book_id'].str.extract('(\d+)', expand=False)
df['book_id'] = df['book_id'].astype(int)
df['pages'] = df['pages'].str.extract('(\d+)', expand=False)
df['pages'] = df['pages'].astype(int)


print(df.loc[df['book_title'] == 'Mein Kampf'])

model = df[['book_id','book_title','author_name','genre_1', 'genre_2', 'num_ratings','num_reviews','pages','publish_date','score']]
print(df.head)

