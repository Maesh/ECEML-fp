"""
Using Stacked Generalization and ensembling to 
get better predictions
"""

import numpy as np
import json as js
import csv
import string
import nltk

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# nltk.data.path.append('/media/maesh/Charming/nltk_data')


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Set random state before keras imports
rs = 19683
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import cross_validation, preprocessing, metrics
import theano
import matplotlib.pyplot as plt

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from foodio import getdata, LemmaTokenizer

def onehots(mat) :
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

def nnk(X,y_uniques,lr=0.1):
	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape
	model.add(Dense(64, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	# model.add(Activation('tanh'))
	model.add(PReLU())
	model.add(Dropout(0.5))
	model.add(Dense(64, init='he_normal',input_dim=32))#, W_regularizer=l2(0.1)))
	# model.add(Activation('tanh'))
	model.add(PReLU())
	model.add(Dropout(0.5))
	model.add(Dense(y_uniques.shape[1], init='he_normal',input_dim=32))#, W_regularizer=l2(0.1)))
	model.add(Activation('softmax'))
	#len(y_uniques)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='categorical_crossentropy', optimizer=sgd)
	return model

def writetest(idx,Xpreds, fil='NN.512.256.64.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([int(idx[row]),Xpreds[row]])

if __name__ == '__main__':
	# Do the training
	res1 = np.genfromtxt('StackGen.DNN.1-2grams.train.csv',delimiter=',')
	res2 = np.genfromtxt('StackGen.DNN.1-2grams.adadelta.train.csv',delimiter=',')
	res3 = np.genfromtxt('StackGen.DNN.1grams.train.csv',delimiter=',')
	res4 = np.genfromtxt('StackGen.NN.1-2grams.train.csv',delimiter=',')
	res5 = np.genfromtxt('StackGen.NN.1grams.train.csv',delimiter=',')

	trainmat = np.mean(np.array([res1,res2,res3,res4,res5]),axis=0)

	# now normalize by row
	# trainmatnorm = preprocessing.normalize(trainmat)

	# pull in true values 
	_, y, _,_,_,_ = getdata(ngram_range=(1,2))

	# Instantiate neural network
	cfr = nnk(trainmat,y,lr=0.1)

	# train it 
	cfr.fit(trainmat, y.toarray(), nb_epoch=25, shuffle=True,
		batch_size=1000, validation_split=0.15,
		show_accuracy=True, verbose=1)

	# now get test values
	tst1 = np.genfromtxt('StackGen.DNN.1-2grams.test.csv',delimiter=',')
	tst2 = np.genfromtxt('StackGen.DNN.1-2grams.adadelta.test.csv',delimiter=',')
	tst3 = np.genfromtxt('StackGen.DNN.1grams.test.csv',delimiter=',')
	tst4 = np.genfromtxt('StackGen.NN.1-2grams.test.csv',delimiter=',')
	tst5 = np.genfromtxt('StackGen.NN.1grams.test.csv',delimiter=',')

	# SVM values need to be vectorized, so define, fit, and transform
	# via CountVectorizer
	unique_cuisines = {'brazilian',
							'british',
							'cajun_creole',
							'chinese',
							'filipino',
							'french',
							'greek',
							'indian',
							'irish',
							'italian',
							'jamaican',
							'japanese',
							'korean',
							'mexican',
							'moroccan',
							'russian',
							'southern_us',
							'spanish',
							'thai',
							'vietnamese'}
	# vect = CountVectorizer(vocabulary=unique_cuisines)
	# svmtst = vect.fit(tst4)
	# svmtst = vect.transform(tst4).toarray()

	# now add all values and normalize
	testmat = np.mean(np.array([tst1,tst2,tst3,tst4,tst5]),axis=0)
	# Xtest = preprocessing.normalize(testmat)

	# finally, we can make predictions on true test set 
	predictions = cfr.predict(testmat)

	# uncomment below if we want to ignore neural net and do simple ensemble
	predictions = tst5
	pred = np.zeros((len(testmat)))
	for row in np.arange(0,len(predictions)) :
		pred[row] = np.argmax(predictions[row])

	unique_cuisines = sorted(list(unique_cuisines))
	newcuisines = []
	for row in np.arange(0,20) :
		newcuisines.append(unique_cuisines[row])

	predstr = []
	for row in np.arange(0,len(predictions)) :
		predstr.append(newcuisines[int(pred[row])])

	test_indices = np.genfromtxt('testing.indices.csv',
						delimiter = ',')
	print("Storing predictions")
	writetest(test_indices,predstr,'StackGen.test.results.meanagg.csv')
