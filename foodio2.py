# -*- coding: utf-8 -*-
"""
A different way of trying the classification 

Special thanks to Dipayan who had a script that this was based on.
"""

from __future__ import print_function
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression


import numpy as np
np.random.seed(19683)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

if __name__ == '__main__':
	# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%
	nltk.data.path.append('/media/maesh/Charming/nltk_data')

	# Create string representations for training data
	traindf = pd.read_json("foodtrain/train.json")
	traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
	traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

	# Create string representations for testing data
	testdf = pd.read_json("foodtest/test.json") 
	testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
	testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

	# Sort by cuisine for memory-based learning
	traindf = traindf.sort('cuisine',ascending=True) 

	# Training corpus of ingredients
	corpustr = traindf['ingredients_string']

	# TFIDF vectorizer for training. 1-grams
	vectorizertr = TfidfVectorizer(stop_words='english',max_features=500,
	                             ngram_range = ( 1 , 1 ),analyzer="word", 
	                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
	tfidftr = vectorizertr.fit_transform(corpustr).todense()
	corpusts = testdf['ingredients_string']
	vectorizerts = TfidfVectorizer(stop_words='english')
	tfidfts = vectorizertr.transform(corpusts)

	X_train = tfidftr

	unique_cuisines_vocab= {'brazilian':0,
							'british':1,
							'cajun_creole':2,
							'chinese':3,
							'filipino':4,
							'french':5,
							'greek':6,
							'indian':7,
							'irish':8,
							'italian':9,
							'jamaican':10,
							'japanese':11,
							'korean':12,
							'mexican':13,
							'moroccan':14,
							'russian':15,
							'southern_us':16,
							'spanish':17,
							'thai':18,
							'vietnamese':19}
	
	vectclasses = CountVectorizer(vocabulary=unique_cuisines_vocab)
	bag_of_classes = vectclasses.fit(traindf['cuisine'])
	y_train = vectclasses.transform(traindf['cuisine']).toarray()

	# y_train = traindf['cuisine']

	X_test = tfidfts


	# #classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
	# parameters = {'C':[1, 10]}
	# #clf = LinearSVC()
	# clf = LogisticRegression()

	# classifier = grid_search.GridSearchCV(clf, parameters)

	# classifier = classifier.fit(predictors_tr,targets_tr)

	# predictions = classifier.predict(predictors_ts)
	# testdf['cuisine'] = predictions
	# testdf = testdf.sort('id' , ascending=True)

	# testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")

	max_features = 39744
	maxlen = 500  # cut texts after this number of words (among top max_features most common words)
	batch_size = 32

	# print(len(X_train), 'train sequences')
	# print(len(X_test), 'test sequences')

	# print("Pad sequences (samples x time)")
	# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	# print('X_train shape:', X_train.shape)
	# print('X_test shape:', X_test.shape)
	# y_train = np.array(y_train)
	# y_test = np.array(y_test)

	print('Build model...')
	model = Graph()
	model.add_input(name='input', input_shape=(maxlen,), dtype=int)
	model.add_node(Embedding(max_features, 32, input_length=maxlen),
	               name='embedding', input='input')
	model.add_node(LSTM(64), name='forward', input='embedding')
	model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
	model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
	model.add_node(Dense(20, activation='softmax'), name='softmax', input='dropout')
	model.add_output(name='output', input='softmax')

	# try using different optimizers and different optimizer configs
	model.compile('adam', {'output': 'categorical_crossentropy'})

	print('Train...')
	model.fit({'input': X_train, 'output': y_train},
				batch_size=batch_size,
				nb_epoch=4)
	acc = accuracy(y_test,
					np.round(np.array(model.predict({'input': X_test},
								batch_size=batch_size)['output'])))
	print('Test accuracy:', acc)