from django.db import models

# Create your models here.
from django.db import models
from django.db.models.fields import DateField

# movie_data_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'url','unknown', 'Action', 'Adventure', 'Animation', "Children's",
#     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

class Movie(models.Model):
    movie_id = models.IntegerField()
    title = models.CharField(max_length=150)
    release_date = models.CharField(default=None,max_length=150)
    video_release_date = models.CharField(default=None,max_length=150)
    url = models.CharField(max_length=500)
    genres = models.CharField(max_length=500)
    language = models.CharField(max_length=150)
    avg_rating = models.FloatField(default=0.0)

    def __str__(self):
        return self.title

class Ratings(models.Model):
    title = models.CharField(max_length=150)
    rating = models.IntegerField(default=0)

    def __str__(self):
        return self.title