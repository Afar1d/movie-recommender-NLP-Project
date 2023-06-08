import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
# Import the necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import pairwise_distances
import math
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# preprocessing file movie
movie = pd.read_csv('movies.csv', sep=";", encoding = "ISO-8859-1")
movie.drop('Unnamed: 3',axis=1,inplace=True)

#Extract year (Year) from title
movie['year'] = movie.title.str.extract('(\(\d\d\d\d\))', expand=False)
#Extract year from (year)
movie['year'] = movie.year.str.extract('(\d\d\d\d)', expand=False)
#remove (year) from title
movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '',regex = True)
#to remove ending whitespace
movie['title'] = movie['title'].apply(lambda x: x.strip())
# convert genres to list and split it
movie['genres'] = movie.genres.str.split('|')
# to save in memory
movie.movieId = movie.movieId.astype('int32')
#fill missing values year 197 NaN
movie.dropna(subset=['year'], inplace=True)
#convert Year to low size
movie.year = movie.year.astype('int16')
#print(movie.head(10))
#print(movie.info())
print(movie.isnull().sum())
print("=============================================================")
##############################################
# preprocessing file rating
rating = pd.read_csv('ratings.csv', sep=";", encoding = "ISO-8859-1")
#rating.drop('timestamp', axis=1, inplace=True)
# NO NULLS
print(rating.isnull().sum())
#print(rating.info())
print("=============================================================")
##############################################
# preprocessing file user
user = pd.read_csv('users.csv', sep=";", encoding = "ISO-8859-1")
#user.drop(['zip-code','occupation'], axis=1, inplace=True)
print(user.isnull().sum())
#print(user.head(10))
print("=============================================================")
##############################################
data = pd.merge(rating,movie)
data = pd.merge(data,user)
#Result.drop(['timestamp','movieId','userId'],axis=1,inplace=True)
#print(data.head(10))

movie['genres'] = movie['genres'].fillna("").astype('str')

#================================================================#




tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movie['genres'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]

# Build a 1-dimensional array with movie titles
titles = movie['title']
indices = pd.Series(movie.index, index=movie['title'])


# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    print("\nRecommendations based on your interest in", title)
    print("These are the top 10 recommended movies: ")
    print(titles.iloc[movie_indices])
    return

recom = genre_recommendations('Toy Story')
