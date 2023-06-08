import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

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
print(rating.info())
print("=============================================================")
##############################################
# preprocessing file user
user = pd.read_csv('users.csv', sep=";", encoding = "ISO-8859-1")
#user.drop(['zip-code','occupation'], axis=1, inplace=True)
print(user.isnull().sum())
print(user.head(10))
print("=============================================================")
##############################################
data = pd.merge(rating,movie)
data = pd.merge(data,user)
#Result.drop(['timestamp','movieId','userId'],axis=1,inplace=True)
print(data.head(10))
print("=============================================================")