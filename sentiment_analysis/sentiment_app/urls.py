from django.conf.urls import url
from . import views

app_name='sentiment_app'

urlpatterns=[
    url(r'^index(?i)/$',views.index,name='index'),
    url(r'^ajax/chk_sentiment(?i)/$', views.chk_sentiment, name='chk_sentiment'),
]