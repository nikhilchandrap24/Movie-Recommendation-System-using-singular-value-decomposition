
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import random
import numpy as np
import pandas as pd
import numpy as np
from typing import *
from scipy.spatial.distance import cosine
from surprise import accuracy
import warnings
import pickle
import pandas as pd    
import matplotlib.pyplot as plt

model = pickle.load(open("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\SVD_model.pkl",'rb'))

def load_movie_dataset() -> pd.DataFrame:
    movie_data_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'url','unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

    movie_data = pd.read_csv("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\u.item", sep = '|', encoding = "ISO-8859-1", header = None, names = movie_data_columns,index_col = 'movie_id')
    movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])
    return movie_data

def load_rating_data() -> pd.DataFrame:
    ratings_data = pd.read_csv("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\u.data",sep = '\t',encoding = "ISO-8859-1",header = None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return ratings_data[['user_id', 'movie_id', 'rating']]

def load_user_data():
  user_data = pd.read_csv("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\u.user",sep = '|', encoding = "ISO-8859-1", header = None,names=['user_id', 'age', 'gender', 'profession','user_score'],index_col = 'user_id')
  return user_data


def coldstart():
    # Solving Cold Start problem using popularity method
    # Calculate mean rating for each movie
    mean_ratings = IMDB_df.groupby('movie_title').mean()['rating']

    # Sort mean ratings in descending order
    mean_ratings_sorted = mean_ratings.sort_values(ascending=False)

    # Get top 100 movies with highest mean rating
    top_10_movies = mean_ratings_sorted.iloc[:10]
    return top_10_movies

ratings_data = load_rating_data()
movies_data = load_movie_dataset()
user_data = load_user_data()
ratings_and_movies = ratings_data.set_index('movie_id').join(movies_data['title']).reset_index()

IMDB_df = ratings_and_movies[['user_id','movie_id','title', 'rating']]
IMDB_df = IMDB_df.rename(columns={'title': 'movie_title'})

# Remove movies with few ratings
movie_ratings = IMDB_df.groupby('movie_title').size()
valid_movies = movie_ratings[movie_ratings > 50]
IMDB_df_filtered = IMDB_df.set_index('movie_title', drop=False).join(valid_movies.to_frame(), how='inner').reset_index(drop=True)
del IMDB_df_filtered[0]

#shuffling the data
IMDB_df_filtered = IMDB_df_filtered.sample(frac=1)
IMDB_df = IMDB_df_filtered

df = IMDB_df

# Define the format of the data using the Reader class
reader = Reader(rating_scale=(1, 5))

# Load the dataframe into a Surprise dataset
data = Dataset.load_from_df(df[['user_id', 'movie_title', 'rating']], reader)

# Get the raw ratings from the data
raw_ratings = data.raw_ratings

# Create a dictionary to map user ids to their ratings for each item
user_item_dict = {}
for uid, iid, rating, _ in raw_ratings:
    if uid not in user_item_dict:
        user_item_dict[uid] = {}
    user_item_dict[uid][iid] = rating

# Convert the dictionary into a pandas dataframe
user_item_df = pd.DataFrame.from_dict(user_item_dict, orient='index')

# Print the resulting user-item matrix
user_item_df.fillna(0)

from numpy.core.function_base import linspace
X =  IMDB_df['movie_id'].unique()
Y = IMDB_df.groupby('movie_title').count()['movie_id'].sort_values(ascending=False)[:10]

Y = Y.to_frame('count')
Y = Y.reset_index()
plt = Y.plot.bar(x = 'movie_title', y = 'count',yticks=[0,200,400,600,800])
plt = plt.figure
plt.savefig('plot.png')



rating_scale = (1, 5)

reader = Reader(rating_scale=rating_scale)

data = Dataset.load_from_df(IMDB_df[['user_id', 'movie_title', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.01)

model = SVD(n_factors=100)
model.fit(trainset)

print("Matrix pu:")
print(model.pu)

print("\nMatrix qi:")
print(model.qi)

print("\nMatrix s:")
# print(model.d)

U = model.pu  # User matrix
V = model.qi  # Item matrix

#normalize the data to compute cosine similarity later

q0_norm_squared = np.sum(V[0] ** 2)
#This line computes the squared L2-norm of the first row of the V matrix and stores it in the variable q0_norm_squared.

V /= np.linalg.norm(V, ord=2, axis=1, keepdims=True)
#This line normalizes each row of the V matrix by dividing it by its L2-norm, 


q0_norm_squared_normalized = np.sum(V[0] ** 2)
#This line computes the squared L2-norm of the first row of the normalized V matrix (which is equivalent to the squared cosine similarity between the first item and all other items) 
#and stores it in the variable q0_norm_squared_normalized.

def display(df: pd.DataFrame):
    item_to_row_idx_df = pd.DataFrame(list(item_to_row_idx.items()), columns=['Movie name', 'V matrix row idx']).set_index('Movie name')
    return item_to_row_idx_df.iloc[:5]

item_to_row_idx: Dict[Any, int] = model.trainset._raw2inner_id_items

display(item_to_row_idx)

toy_story_row_idx : int = item_to_row_idx['Toy Story (1995)']



def get_vector_by_movie_title(movie_title: str, trained_model: SVD) -> np.array:
    """Returns the latent features of a movie in the form of a numpy array"""
    item_to_row_idx = trained_model.trainset._raw2inner_id_items
    movie_row_idx = item_to_row_idx[movie_title]
    return trained_model.qi[movie_row_idx]

def cosine_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Returns a float indicating the similarity between two vectors"""
    return 1-cosine(vector_a, vector_b)

movie_a = 'Toy Story (1995)'
movie_b = 'Jurassic Park (1993)'
vector_a = get_vector_by_movie_title(movie_a, model)
vector_b = get_vector_by_movie_title(movie_b, model)
similarity_score = 1-cosine_distance(vector_a, vector_b)
print(f"The similarity score between '{movie_a}' and '{movie_b}' is: {similarity_score:.4f}")

toy_story_vec = get_vector_by_movie_title('Toy Story (1995)', model)
wizard_of_oz_vec = get_vector_by_movie_title('Wizard of Oz, The (1939)', model)

similarity_score = 1-cosine_distance(toy_story_vec, wizard_of_oz_vec)

starwars_idx = model.trainset._raw2inner_id_items['Star Wars (1977)']
roj_idx = model.trainset._raw2inner_id_items['Return of the Jedi (1983)']
aladdin_idx = model.trainset._raw2inner_id_items['Aladdin (1992)']

starwars_vector = model.qi[starwars_idx]
return_of_jedi_vector = model.qi[roj_idx]
aladdin_vector = model.qi[aladdin_idx]

cosine_distance(starwars_vector, return_of_jedi_vector)

cosine_distance(starwars_vector, aladdin_vector)

def display(similarity_table):
    similarity_table = pd.DataFrame(
        similarity_table,
        columns=['vector cosine distance', 'movie title']
    ).sort_values('vector cosine distance', ascending=False)
    return similarity_table.iloc[:4]

def get_top_similarities(movie_title, model) -> pd.DataFrame:
    """
    Returns a DataFrame of the top 4 most similar movies to a given movie title,
    based on cosine similarity between their latent feature vectors.
    """
    
    movie_vector: np.array = get_vector_by_movie_title(movie_title, model)
    similarity_table = []
    
    for other_movie_title in model.trainset._raw2inner_id_items.keys():
        other_movie_vector = get_vector_by_movie_title(other_movie_title, model)
        
        similarity_score =  cosine_distance(other_movie_vector, movie_vector)
        similarity_table.append((similarity_score, other_movie_title))
    
    return pd.DataFrame(
        sorted(similarity_table,reverse = True)[1:10],
        columns=['vector cosine distance', 'movie title']
    )

print(get_top_similarities('Star Wars (1977)', model))

def multipleMovieRecommendation(user_input): 
    movie_to_idx = {}
    for i, row in movies_data.iterrows():
        movie_to_idx[row['title']] = i

    user_df = pd.DataFrame.from_dict(user_input, orient='index', columns=['rating'])

    unseen_movies = movies_data[~movies_data['title'].isin(user_df.index)]

    unseen_ratings = pd.merge(unseen_movies, ratings_data, on='movie_id')[['title', 'user_id', 'rating']]

    unseen_ratings['predicted_rating'] = unseen_ratings.apply(lambda x: model.predict(x['user_id'], movie_to_idx[x['title']]).est, axis=1)

    recommendations = unseen_ratings.sort_values('predicted_rating', ascending=False).head(10)['title'].tolist()
    return recommendations

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
print("RMSE:", rmse)

mae = accuracy.mae(predictions)
print("MAE:", mae)
