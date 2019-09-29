from django.shortcuts import render
from django.shortcuts import render
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import gc
import os
import pickle
from nltk.corpus import wordnet
from django.http import JsonResponse
gc.collect()

def chk_sentiment(request):
    data=[] 
    data.append(request.GET.get('review', None))
    X_live=preprocessing_text(data)
    mean_3_clfs_mnb=run_multinomial_nb(X_live)
    mean_3_clfs_xgb=run_xgboost(X_live)
    mean_3_clfs_ada=run_adaboost(X_live)
    mean_3_clfs_et=run_extratree(X_live)
    all_clfs_data = np.column_stack((mean_3_clfs_mnb, mean_3_clfs_ada, mean_3_clfs_xgb,mean_3_clfs_et))
    data_predicted=final_prediction(all_clfs_data)
    if data_predicted[0]==1:
        data = {
        'sentiment':'Positive'
                }
    else:
        data = {
            'sentiment':'Negative'
        }
    return JsonResponse(data)


def index(request):   
    page=""
    url="https://www.tripadvisor.in/Restaurant_Review-g297628-d6117364-Reviews-Taco_Bell-Bengaluru_Bangalore_District_Karnataka.html"
    if request.method == 'GET':
        page  = requests.get(url)  
    else:
        url = request.POST['url']
        page  = requests.get(url)
    
    context=do_analysis(page,url)
    return render(request,r'sentiment_app/index.html',context)

def do_analysis(page,url):
    list_review=[]
    data=[]
    soup = BeautifulSoup(page.content, 'html.parser')
    for item in soup.find_all('p', class_='partial_entry'):        
            data.append(item.get_text())
    X_live=preprocessing_text(data)
    mean_3_clfs_mnb=run_multinomial_nb(X_live)
    mean_3_clfs_xgb=run_xgboost(X_live)
    mean_3_clfs_ada=run_adaboost(X_live)
    mean_3_clfs_et=run_extratree(X_live)
    all_clfs_data = np.column_stack((mean_3_clfs_mnb, mean_3_clfs_ada, mean_3_clfs_xgb,mean_3_clfs_et))
    data_predicted=final_prediction(all_clfs_data)

    for i in range(0,len(data)):
        review={}
        review['review']=data[i]
        if data_predicted[i]==0:            
            review['sentiment']="Negative"
        else:
            review['sentiment']="Positive"
        list_review.append(review)
    
    restaurantname=url.split('Reviews-')[1]
    restaurantname=restaurantname.split('-')[0]
    context = {
                'reviews':list_review,
                'restaurantname':restaurantname
            }
    return context

def preprocessing_text(data_list):
    live_data = pd.DataFrame()
    live_data["review"]=data_list
    upper_words=no_of_Uppercase_words(live_data)
    live_data['upper_case_words']=upper_words
    live_data=remove_blanks(live_data)
    live_data=remove_hrml_urls(live_data)
    live_data=convert_to_lowercase(live_data)
    live_data=removing_punctuation(live_data)
    live_data=lemmatization(live_data)
    live_data=chk_english_word(live_data)

    newDF_live = pd.DataFrame()
    newDF_live['review']=live_data['review']

    cv=load_from_pikl("D:\Python\sentiment_analysis_web_app\sentiment_proj\models\cv_pikl.pkl")
    X_live=cv.transform(newDF_live['review'])
    X_live=X_live.todense()
    X_live=np.c_[ X_live, live_data['upper_case_words'] ]
    return X_live

def run_multinomial_nb(X_live):
    data_3_clfs = np.empty((3, len(X_live)))
    clf=load_from_pikl("D:\Python\sentiment_analysis_web_app\sentiment_proj\models\mnb1.pkl")
    data_3_clfs[0, :] = clf.predict(X_live)

    clf=load_from_pikl("D:\Python\sentiment_analysis_web_app\sentiment_proj\models\mnb2.pkl")
    data_3_clfs[1, :] = clf.predict(X_live)

    clf=load_from_pikl("D:\Python\sentiment_analysis_web_app\sentiment_proj\models\mnb3.pkl")
    data_3_clfs[2, :] = clf.predict(X_live)

    mean_3_clfs_mnb=np.zeros((len(X_live),))
    mean_3_clfs_mnb[:] = data_3_clfs.mean(axis=0)
    arr = np.array([.4, 2, 3])
    lmd_opr=lambda x: 1 if x>=.5 else 0

    for i in range(0,len(X_live)):
        if mean_3_clfs_mnb[i] >=.5 :
            mean_3_clfs_mnb[i]=1
        else:
            mean_3_clfs_mnb[i]=0
    return mean_3_clfs_mnb

def run_xgboost(X_live):
    data_3_clfs = np.empty((3, len(X_live)))
    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\xgb1.pkl")
    data_3_clfs[0, :] = clf.predict(X_live)

    clf=load_from_pikl(r'''D:\Python\sentiment_analysis_web_app\sentiment_proj\models\xgb2.pkl''')
    data_3_clfs[1, :] = clf.predict(X_live)

    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\xgb3.pkl")
    data_3_clfs[2, :] = clf.predict(X_live)

    mean_3_clfs_xgb=np.zeros((len(X_live),))
    mean_3_clfs_xgb[:] = data_3_clfs.mean(axis=0)


    for i in range(0,len(X_live)):
        if mean_3_clfs_xgb[i] >=.5 :
            mean_3_clfs_xgb[i]=1
        else:
            mean_3_clfs_xgb[i]=0
    return mean_3_clfs_xgb


def run_adaboost(X_live):
    data_3_clfs = np.empty((3, len(X_live)))
    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\adaboost1.pkl")
    data_3_clfs[0, :] = clf.predict(X_live)

    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\adaboost2.pkl")
    data_3_clfs[1, :] = clf.predict(X_live)

    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\adaboost3.pkl")
    data_3_clfs[2, :] = clf.predict(X_live)

    mean_3_clfs_ada=np.zeros((len(X_live),))
    mean_3_clfs_ada[:] = data_3_clfs.mean(axis=0)


    for i in range(0,len(X_live)):
        if mean_3_clfs_ada[i] >=.5 :
            mean_3_clfs_ada[i]=1
        else:
            mean_3_clfs_ada[i]=0
    return mean_3_clfs_ada


def run_extratree(X_live):
    data_3_clfs = np.empty((3, len(X_live)))
    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\et1.pkl")
    data_3_clfs[0, :] = clf.predict(X_live)

    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\et2.pkl")
    data_3_clfs[1, :] = clf.predict(X_live)

    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\et3.pkl")
    data_3_clfs[2, :] = clf.predict(X_live)

    mean_3_clfs_et=np.zeros((len(X_live),))
    mean_3_clfs_et[:] = data_3_clfs.mean(axis=0)

    for i in range(0,len(X_live)):
        if mean_3_clfs_et[i] >=.5 :
            mean_3_clfs_et[i]=1
        else:
            mean_3_clfs_et[i]=0
    return mean_3_clfs_et


def final_prediction(all_clfs_data):
    clf=load_from_pikl(r"D:\Python\sentiment_analysis_web_app\sentiment_proj\models\finalclf_xgb.pkl")
    return clf.predict(all_clfs_data)

def no_of_Uppercase_words(data_frame):
    """Anger or rage is quite often expressed by writing in UPPERCASE words
    which makes this a necessary operation to identify those words."""
    upper_words = data_frame['review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    return upper_words

def remove_hrml_urls(data_frame):
    """removes html/url links"""
    data_frame['review']=data_frame['review'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))
    return data_frame

def remove_blanks(data_frame):
    data_frame['review'].replace('  ', np.nan, inplace=True)
    #data_frame['review'].replace(' ', np.nan, inplace=True)
    data_frame= data_frame.dropna(subset=['review'])
    return data_frame

def remove_hrml_urls(data_frame):
    """removes html/url links"""
    data_frame['review']=data_frame['review'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))
    return data_frame

def convert_to_lowercase(data_frame):
    """The first pre-processing step which we will do is transform our tweets 
    into lower case. This avoids having multiple copies of the same words. For example, while calculating the word count
    ,‘Analytics’ and ‘analytics’ will be taken as different words."""
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data_frame

def removing_punctuation(data_frame):
    """The next step is to remove punctuation, as it doesn’t add any extra information while 
    treating text data. Therefore removing all instances of 
    it will help us reduce the size of the training data."""
    data_frame['review']  =data_frame['review'] .str.replace('[^\w\s]','')
    return data_frame

def lemmatization(data_frame):
    """Lemmatization is a more effective option than stemming because it converts 
    the word into its root word, rather than just stripping the suffices. 
    It makes use of the vocabulary and does a morphological analysis to obtain the root word. 
    Therefore, we usually prefer using lemmatization over stemming."""   
    from textblob import Word
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data_frame

def is_english_word(word):
    chk=wordnet.synsets(word)
    if len(chk) ==0:
        return ""
    else:
        return word

def chk_english_word(data_frame):
    data_frame['review'] = data_frame['review'].apply(lambda x: " ".join([is_english_word(word) for word in x.split()]))
    return data_frame

def load_from_pikl(name):   
    with open(name, 'rb') as fid:        
        data=pickle.load(fid)
    fid.close
    return data