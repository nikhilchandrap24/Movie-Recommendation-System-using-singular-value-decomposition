from django.contrib import admin
from django.urls import path, include

from user_interface import views

urlpatterns = [
    path('',views.home,name = 'home'),
    path('home/',views.home,name = 'home'),
    path('search/',views.search,name = 'search'),
    path('rateMovie/',views.rateMovie,name='rateMovie'),
    path('recommendations/',views.recommend,name='recommendations'),
    path('deleteAll/',views.deleteAll,name='deleteAll'),
    path('addMovie/',views.addMovie,name='addMovie'),
    path('delete/<str:title>',views.delete,name='delete'),
    path('about/',views.about,name='about'),
    ]