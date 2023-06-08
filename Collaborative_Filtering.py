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



#================================================================#
train_data, test_data = train_test_split(data, test_size=0.2)
train_data_matrix = train_data[['userId', 'movieId', 'rating']]
test_data_matrix = test_data[['userId', 'movieId', 'rating']]

num_users_train = train_data_matrix['userId'].unique().max()
num_movies_train = train_data_matrix['movieId'].unique().max()

num_users_test = test_data_matrix['userId'].unique().max()
num_movies_test = test_data_matrix['movieId'].unique().max()

train_matrix = np.zeros((num_users_train, num_movies_train))
for i in train_data_matrix.itertuples():
    train_matrix[i[1]-1,i[2]-1] = i[3]

test_matrix = np.zeros((num_users_test, num_movies_test))
for i in test_data_matrix.itertuples():
    test_matrix[i[1]-1,i[2]-1] = i[3]


u, s, vt = svds(train_matrix, k = 20)

s_matrix = np.diag(s)
predictions_svd = np.dot(np.dot(u,s_matrix),vt)

def recom(user_id, train_matrix):
    user_rating = predictions_svd[user_id-1,:]
    train_indices = np.where(train_matrix[user_id-1,:] == 0)[0]
    user_recommendations = user_rating[train_indices]

    print('\nRecommendations for user {} : '.format(user_id))
    for movie_id in user_recommendations.argsort()[-10:][: : -1]:
        print(np.array(movie.loc[movie['movieId'] == movie_id+1].title))
    return

recom(user_id=886, train_matrix=train_matrix)

