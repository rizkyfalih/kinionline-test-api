import json, falcon, re, pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class DecisionModelRequest():

    def on_post(self, req, res):
        res.status = falcon.HTTP_200
        data = json.loads(req.stream.read())

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

        # Build a vocab
        vectorizer = CountVectorizer(max_features=6897)
        dataFeature = vectorizer.fit(dataTrain)

        # Clean the data
        corpus = str(data['text'])
        corpus = re.sub('[^a-zA-Z]', ' ', corpus)
        corpus = corpus.lower()
        corpus = corpus.split()
        corpus = ' '.join(corpus)
        listCorpus = []
        listCorpus.append(corpus)

        # Feature Extraction
        X = vectorizer.transform(listCorpus).toarray()

        # Load the model
        filename_model = 'decision_model.sav'
        decision_model = pickle.load(open(filename_model, 'rb'))
        predict = decision_model.predict(X)

        # The output
        if(predict[0] == 1):
            pred_sentiment = 'Postif'
        else:
            pred_sentiment = 'Negatif'

        output = {
            'sentiment' : pred_sentiment
        }

        res.body = json.dumps(output)

class RanforestModelRequest():

    def on_post(self, req, res):
        res.status = falcon.HTTP_200
        data = json.loads(req.stream.read())

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

        # Build a vocab
        vectorizer = CountVectorizer(max_features=6897)
        dataFeature = vectorizer.fit(dataTrain)

        # Clean the data
        corpus = str(data['text'])
        corpus = re.sub('[^a-zA-Z]', ' ', corpus)
        corpus = corpus.lower()
        corpus = corpus.split()
        corpus = ' '.join(corpus)
        listCorpus = []
        listCorpus.append(corpus)

        # Feature Extraction
        X = vectorizer.transform(listCorpus).toarray()

        # Load the model
        filename_model = 'ranforest_model.sav'
        decision_model = pickle.load(open(filename_model, 'rb'))
        predict = decision_model.predict(X)

        # The output
        if(predict[0] == 1):
            pred_sentiment = 'Postif'
        else:
            pred_sentiment = 'Negatif'

        output = {
            'sentiment' : pred_sentiment
        }

        res.body = json.dumps(output)

api = falcon.API()
api.add_route('/decision',DecisionModelRequest())
api.add_route('/ranfost',RanforestModelRequest())
