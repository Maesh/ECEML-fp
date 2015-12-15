import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
import string 

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


def nnk(X,y_uniques,lr=0.1):
	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape
	model.add(Dense(512, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	# model.add(PReLU())
	model.add(Dropout(0.5))
	model.add(Dense(256, init='he_normal',input_dim=512))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	# model.add(PReLU())
	model.add(Dropout(0.5))
	model.add(Dense(64, init='he_normal',input_dim=256))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	# model.add(PReLU())
	model.add(Dropout(0.5))
	model.add(Dense(len(y_uniques), init='he_normal',input_dim=64))#, W_regularizer=l2(0.1)))
	model.add(Activation('softmax'))
	#len(y_uniques)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='categorical_crossentropy', optimizer=sgd)
	return model

def rnnkeras(X,y_uniques,lr=0.1) :
	# input dimension should be (nb_samples, timesteps, input_dim)

	model = Sequential()
	model.add(Embedding(X,3))
	model.add(SimpleRNN(100,input_dim=X,init='he_normal'))
	# model.add(Activation('tanh'))
	# model.add(Dropout(0.5))
	# model.add(SimpleRNN(256,init='he_normal',input_dim=256))
	# model.add(Activation('tanh'))
	# model.add(Dropout(0.5))
	model.add(Dense(y_uniques,init='he_normal',input_dim=100))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd)
	return model

def writetest(idx,Xpreds, fil='NN.512.256.64.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([int(idx[row]),Xpreds[row]])

def writestackgen(Xpreds, fil='NN.512.256.64.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow(Xpreds[row])

if __name__ == '__main__':
	print("Importing Data")
	# these will be sparse
	# X, y, unique_cuisines,classes,test_indices,Xtest = getdata() # import the data

	# # Split into training and validation sets
	# rs = 19683
	# X_train, X_test, y_train, y_test = \
	# 	cross_validation.train_test_split(X, y, \
	# 		test_size=0.4, random_state=rs)

	# print("Training classifier")
	# # Train the classifier and fit to training data
	# clf2 = nnk(X,unique_cuisines,lr=0.1)
	# # # clf2 = rnnkeras(39744,20,lr=0.1)
	# f = clf2.fit(X_train, y_train, nb_epoch=25, shuffle=True,
	# 	batch_size=1000, validation_split=0.15,
	# 	show_accuracy=True, verbose=1)

	# print("Making predictions on validation set")
	# Make predictions on validation data
	# predictions = clf2.predict(X_test, batch_size=100, verbose=1)

	# Take max value in preds rows as classification
	# pred = np.zeros((len(X_test)))
	# yint = np.zeros((len(X_test)))
	# for row in np.arange(0,len(predictions)) :
	# 	pred[row] = np.argmax(predictions[row])
	# 	yint[row] = np.argmax(y_test[row])

	# print("Classifier Accuracy = %d"%(metrics.accuracy_score(yint,pred)))

	
	#####
	# now to test
	#####
	print("Testing classifier on Test data")
	print("Re-train with full training set")
	X, y, unique_cuisines,classes,test_indices,_ = getdata(ngram_range=(1,2)) # import the data

	clf2 = nnk(X,unique_cuisines,lr=0.1)
	f = clf2.fit(X.tocsc(), y.tocsc(), nb_epoch=25, batch_size=100, 
		validation_split=0.15, show_accuracy=True)

	predictions = clf2.predict(X.tocsc(), batch_size=100, verbose=1)
	writestackgen(predictions,'StackGen.NN.1-2grams.train.csv')

	_, _, unique_cuisines,classes,test_indices,Xtest = getdata(ngram_range=(1,2))
	predictions = clf2.predict(Xtest.toarray(), batch_size=100, verbose=1)
	writestackgen(predictions,'StackGen.NN.1-2grams.test.csv')
	# # Take max value in preds rows as classification
	# pred = np.zeros((len(Xtest)))
	# for row in np.arange(0,len(predictions)) :
	# 	pred[row] = np.argmax(predictions[row])

	# unique_cuisines = {'brazilian',
	# 						'british',
	# 						'cajun_creole',
	# 						'chinese',
	# 						'filipino',
	# 						'french',
	# 						'greek',
	# 						'indian',
	# 						'irish',
	# 						'italian',
	# 						'jamaican',
	# 						'japanese',
	# 						'korean',
	# 						'mexican',
	# 						'moroccan',
	# 						'russian',
	# 						'southern_us',
	# 						'spanish',
	# 						'thai',
	# 						'vietnamese'}
	# unique_cuisines = sorted(list(unique_cuisines))
	# newcuisines = []
	# for row in np.arange(0,20) :
	# 	newcuisines.append(unique_cuisines[row])

	# predstr = []
	# for row in np.arange(0,len(predictions)) :
	# 	predstr.append(newcuisines[int(pred[row])])

	# test_indices = np.genfromtxt('testing.indices.csv',
	# 					delimiter = ',')
	# print("Storing predictions")
	# writetest(test_indices,predstr,'NN.512.256.64.PReLU.5000feats.csv')

