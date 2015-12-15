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

from foodio import getdata, writedata, LemmaTokenizer

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


if __name__ == '__main__':
	# Do the training
	res1 = np.genfromtxt('StackGen.DNN.1-2grams.train.csv',delimiter=',')
	res2 = np.genfromtxt('StackGen.DNN.1grams.train.csv',delimiter=',')
	res3 = np.genfromtxt('StackGen.NN.1-2grams.train.csv',delimiter=',')
	res4 = np.genfromtxt('StackGen.SVM.1-2grams.train.csv',delimiter=',')

	trainmat = np.sum(np.array([res1,res2,res3]),axis=0)

	# now normalize by row
	trainmatnorm = preprocessing.normalize(trainmat)

	# pull in true values
	y_test = np.genfromtxt('StackGen.train.actualvalues.csv',delimiter=',')

	# Instantiate neural network
	cfr = nnk(trainmatnorm,y_test,lr=0.1)

	# train it 
	cfr.fit(trainmatnorm, y_test, nb_epoch=25, shuffle=True,
		batch_size=1000, validation_split=0.15,
		show_accuracy=True, verbose=1)