from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='digit_recognition-home'),
    path('upload/', views.upload, name='digit_recognition-upload'),
]
