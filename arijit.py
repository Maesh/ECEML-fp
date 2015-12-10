import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb

# Set random state before keras imports
rs = 19683
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import cross_validation, preprocessing, metrics
import numpy as np
import theano
import matplotlib.pyplot as plt


from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def nnk(X,y_uniques,lr=0.1):
	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape
	model.add(Dense(512, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(256, init='he_normal',input_dim=512))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='he_normal',input_dim=256))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(len(y_uniques), init='he_normal',input_dim=64))#, W_regularizer=l2(0.1)))
	model.add(Activation('softmax'))
	#len(y_uniques)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='binary_crossentropy', optimizer=sgd)
	return model
	# Batch size = 100 seems to have stabilized it
	#model.fit(X_train, y_train, nb_epoch=100, batch_size=1000)
	# score = model.evaluate(X_test, y_test, batch_size=1000)
	# preds = model.predict(X_test, batch_size=1000, verbose=1)
	# print score

if __name__ == '__main__':
	
	with open('foodtrain/train.json') as json_data:
		data = js.load(json_data)
		json_data.close()

	classes = [item['cuisine'] for item in data]
	ingredients = [item['ingredients'] for item in data]
	unique_ingredients = set(item for sublist in ingredients for item in sublist)
	unique_cuisines = set(classes)

	# print len(data)
	# print (classes)
	print "Number of recipes = %d"%(len(ingredients))
	print "Number of unique ingredients = %d"%(len(unique_ingredients))
	print "Number of unique cuisines = %d"%(len(unique_cuisines))

	X = np.zeros((len(ingredients), len(unique_ingredients)))

	# Compile feature matrix. Could be sparse but dense is fine for us.
	# Each feature is an ingredient. Each row is a recipe. For each 
	# column (feature), a 1 indicates the recipe has that ingredient,
	# while a 0 indicates it does not.
	for d,dish in enumerate(ingredients):
		for i,ingredient in enumerate(unique_ingredients):
			if ingredient in dish:
				X[d,i] = 1

	# Also need to ensure
	y=np.zeros((len(classes),len(unique_cuisines)))
	for c,clas in enumerate(classes):
		for q,cuisine in enumerate(unique_cuisines):
			if cuisine in clas:
				y[c,q] = 1

	# Split into training and validation sets
	rs = 19683
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.4, random_state=rs)

	# Train the classifier and fit to training data
	clf2 = nnk(X_train,unique_cuisines,lr=0.1)
	f = clf2.fit(X_train, y_train, nb_epoch=30, batch_size=100, validation_split=0.15)

	# Make predictions on validation data
	predictions = clf2.predict(X_test, batch_size=100, verbose=1)

	# Take max value in preds rows as classification
	pred = np.zeros((len(X_test)))
	yint = np.zeros((len(X_test)))
	for row in np.arange(0,len(predictions)) :
		pred[row] = np.argmax(predictions[row])
		yint[row] = np.argmax(y_test[row])

	accs.append(metrics.accuracy_score(yint,pred))

	# model = Sequential()
	# model.add(Dense(512, input_dim=X.shape[1]))
	# model.add(PReLU())
	# model.add(BatchNormalization())
	# model.add(Dropout(0.5))

	# model.add(Dense(512))
	# model.add(PReLU())
	# model.add(BatchNormalization())
	# model.add(Dropout(0.5))

	# model.add(Dense(512))
	# model.add(PReLU())
	# model.add(BatchNormalization())
	# model.add(Dropout(0.5))

	# model.add(Dense(len(unique_cuisines)))
	# model.add(Activation('softmax'))

	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(loss='categorical_crossentropy', optimizer=sgd)

	# print('Training model...')
	# model.fit(X_train, y_train, nb_epoch=200, batch_size=128, validation_split=0.15)
	# predictionsDN = model.predict(X_test)
	# # Take max value in preds rows as classification
	# predDN = np.zeros((len(X_test)))
	# yintDN = np.zeros((len(X_test)))
	# for row in np.arange(0,len(predictionsDN)) :
	# 	predDN[row] = np.argmax(predictionsDN[row])
	# 	yintDN[row] = np.argmax(y_test[row])
	# print metrics.accuracy_score(yintDN,predDN)