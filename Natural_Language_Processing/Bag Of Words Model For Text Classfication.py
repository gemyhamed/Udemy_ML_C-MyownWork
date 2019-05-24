# -*- coding: utf-8 -*-
"""
Created on Fri May 24 01:37:12 2019

@author: Gemy
"""

import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split




df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t')
ps = PorterStemmer()
stop_words = stopwords.words('english')


clean = []
for r in df['Review']:
    r.lower()
    r = re.sub('[^a-zA-Z]'," ",r)
    r = word_tokenize(r)
    r = [x for x in r if x not in stop_words]
    r = [ps.stem(x) for x in r ]
    r = " ".join(r)
    
    clean.append(r)
    
    
# put all the clean reviews into a dataframe

df2= pd.DataFrame(clean , columns = ['reviews'])
df2['likes'] = df['Liked']


# Define X - Y 

x = df2['reviews']
y = df2['likes']

# Create tf-idf Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_tf = tf.fit_transform(x).toarray()

# Train Logistic Regression Model
x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tf, y, test_size=0.2, random_state=0)
log_tv = LogisticRegression() 
log_tv.fit(x_train_tv,y_train_tv)
y_pred_tv = log_tv.predict(x_test_tv)
print(confusion_matrix(y_test_tv,y_pred_tv))
print(classification_report(y_test_tv,y_pred_tv))

# Train Naive Bayes Model
x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tf, y, test_size=0.2, random_state=0)
log_tv = GaussianNB() 
log_tv.fit(x_train_tv,y_train_tv)
y_pred_tv = log_tv.predict(x_test_tv)
print(confusion_matrix(y_test_tv,y_pred_tv))
print(classification_report(y_test_tv,y_pred_tv))
