from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from .models import Movie, Ratings
from .SVDmodel import *

model = pickle.load(open("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\SVD_model.pkl",'rb'))

def load_movie_dataset():
    movie_data_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'url','unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

    movie_data = pd.read_csv("E:\\VIII Sem\\Project-UI\\movie_recommendation_system\\user_interface\\u.item", sep = '|', encoding = "ISO-8859-1", header = None, names = movie_data_columns)
    movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])


    return movie_data



def dataset_to_database():
    df = load_movie_dataset()
    df_records = df.to_dict('records')[1:]

    model_instances = []
    print(df_records[0])
    for record in df_records : 
        
        movie_id = record['movie_id']
        title = record['title']

        if record['release_date'] != 'nan':
            release_date =  record['release_date']
        else:
            release_date = ""

        if record['video_release_date'] != 'nan':
            video_release_date = record['video_release_date']
        else:
            video_release_date = ""

        if record['url'] != 'nan':
            url = record['url']
        else:
            url = ""
        genres = ""
        for genre in ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']:
            if record[genre] == 1:
                genres += (genre + ', ')
        genres = genres[:-2]
        print(movie_id,title,release_date,video_release_date,url,genres)
        model_instances.append(Movie(movie_id=movie_id,title=title,release_date=release_date,video_release_date=video_release_date,url=url,genres=genres))

    Movie.objects.bulk_create(model_instances)


def home(request):
    # objs = Movie.objects.all()
    # objs.delete()
    # dataset_to_database()
    return render(request,'home.html')

def addMovie(request):
    all_ratings = Ratings.objects.all()
    if request.method == 'POST':
        titles = [x.title for x in all_ratings]
        movie = request.POST['movie_title']
        rating = request.POST['rating']
        print(movie,rating)
        if movie not in titles:
            obj = Ratings(title=movie,rating=rating)
            obj.save()
        movies = Movie.objects.all()
        all_ratings = Ratings.objects.all()
        return render(request,'search.html',{'movies': movies,'all_ratings': all_ratings})


def search(request):

    print('Into search method')
    movies = Movie.objects.all()
    all_ratings = Ratings.objects.all()
    if request.method == 'POST':
        print('Into post method')
        return render(request,'recommend.html',{'movies': movies,'all_ratings': all_ratings})
    else:
        all_ratings = Ratings.objects.all()
        return render(request,'search.html',{'movies': movies,'all_ratings': all_ratings})

def deleteAll(request):
    print("inside deleteAll")
    obj =  Ratings.objects.all()  
    obj.delete()  
    all_ratings = Ratings.objects.all()
    movies = Movie.objects.all()
    return render(request,'search.html',{'movies': movies,'all_ratings': all_ratings})

def delete(request,title):
    obj = Ratings.objects.filter(title=title)
    obj.delete()
    all_ratings = Ratings.objects.all()
    movies = Movie.objects.all()
    return render(request,'search.html',{'movies': movies,'all_ratings': all_ratings})

def rateMovie(request):
    all_ratings = Ratings.objects.all()    
    movies = Movie.objects.all()
    return render(request,'rateMovie.html',{'movies': movies,'all_ratings': all_ratings})

def recommend(request):
    all_ratings = Ratings.objects.all()
    if request.method == 'POST':
        d={}
        if len(all_ratings) == 0:
            recommendations = coldstart()
            recommendations = pd.DataFrame(recommendations).transpose()
        elif len(all_ratings) == 1:
            recommendations = get_top_similarities(all_ratings[0].title,model)
            recommendations = recommendations['movie title'].tolist()
        else:
            for movie in all_ratings:
                d[movie.title] = movie.rating
            recommendations = multipleMovieRecommendation(d)
    else:
        d={}
        if len(all_ratings) == 0:
            recommendations = coldstart()
            recommendations = pd.DataFrame(recommendations).transpose()
        elif len(all_ratings) == 1:
            recommendations = get_top_similarities(all_ratings[0].title,model)
            recommendations = recommendations['movie title'].tolist()
        else:
            for movie in all_ratings:
                d[movie.title] = movie.rating
            recommendations = multipleMovieRecommendation(d)
        print(recommendations)
    return render(request,'recommendations.html',{'recommendations': recommendations})

def about(request):
    return render(request,'about.html')

