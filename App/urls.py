from django.urls import path
from App import views

urlpatterns = [
    path('', views.index, name='indexpage'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('prediction/', views.prediction, name='prediction'),
]
