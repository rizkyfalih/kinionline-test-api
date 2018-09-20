# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:25:20 2018

@author: rizkyfalih
"""

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def cleaned_text(data):
    newCorpus = []
    for i in data.index:
        corpus = re.sub('[^a-zA-Z]', ' ', data.loc[i, 'text'])
        corpus = corpus.lower()
        corpus = corpus.split()
        corpus = ' '.join(corpus)
        newCorpus.append(corpus)
    return newCorpus

def bag_of_word(corpus):
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    Y = cv.get_feature_names()

    return X,Y

def model_DecisionTree(X_train,y_train):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train,y_train)

    filename = 'decision_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def model_RandomForest(X_train,y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)

    filename = 'ranforest_model.sav'
    pickle.dump(model, open(filename, 'wb'))

f_neg = open('neg.txt', 'r')
data_neg = f_neg.read()
data_neg = data_neg.split('\n')

f_pos = open('pos.txt', 'r')
data_pos = f_pos.read()
data_pos = data_pos.split('\n')

dataTrain = []
for i in range(1000):
    dataTrain.append(data_neg[i])

for i in range(1000):
    dataTrain.append(data_pos[i])


y = []
for i in range(1000):
     y.append(0)

for i in range(1000):
     y.append(1)


# Create dataFrame
df = pd.DataFrame(dataTrain, columns=['text'])
df['sentiment'] = y

corpus = cleaned_text(df)
X, list_word = bag_of_word(corpus)
y = df.iloc[:, 1].values

y0=np.array(np.where(y==0)).ravel()
y1=np.array(np.where(y==1)).ravel()

X_train = np.vstack((X[y1[:950],:],X[y0[:950],:]))
X_test = np.vstack((X[y1[950:],:],X[y0[950:],:]))

y_train = np.hstack((y[y1[:950]],y[y0[:950]]))
y_test = np.hstack((y[y1[950:]],y[y0[950:]]))

# model_DecisionTree(X_train, y_train)
# model_RandomForest(X_train, y_train)

filename_decision = 'decision_model.sav'
decision_model = pickle.load(open(filename_decision, 'rb'))
result = decision_model.score(X_test, y_test)
predict = decision_model.predict(X_test)
print(result)

filename_ranforest = 'ranforest_model.sav'
ranforest_model = pickle.load(open(filename_ranforest, 'rb'))
result = ranforest_model.score(X_test, y_test)
predict = ranforest_model.predict(X_test)
print(result)
