import csv as csv
import scipy as scipy
import numpy as np
import json as js
import string
# Set random state before keras imports
rs = 19683
from sklearn import svm, cross_validation, preprocessing, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from foodio import getdata, writedata, LemmaTokenizer

def writetest(idx,Xpreds, fil='svm1.csv') :
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

	print("Loading Data")
	#Training
	with open('foodtrain/train.json') as json_data:
		data = js.load(json_data)
		json_data.close()

	with open('foodtest/test.json') as json_data:
		testdata = js.load(json_data)
		json_data.close()

	print("Data loaded")
	classes = [item['cuisine'] for item in data]
	ingredients = [item['ingredients'] for item in data]
	test_indices = [item['id'] for item in testdata]

	test_ingredients = [item['ingredients'] for item in testdata]

	print("Cleaning text data")
	# Text cleaning
	for row in np.arange(0,len(ingredients)) :
		for col in np.arange(0,len(ingredients[row])) :
			temp = filter(lambda x: x in string.printable, ingredients[row][col])
			ingredients[row][col] = str(temp)

	for row in np.arange(0,len(test_ingredients)) :
		for col in np.arange(0,len(test_ingredients[row])) :
			temp = filter(lambda x: x in string.printable, test_ingredients[row][col])
			test_ingredients[row][col] = str(temp)

	for row in np.arange(0,len(classes)) :
		temp = filter(lambda x: x in string.printable, classes[row])
		classes[row] = str(temp)

	print("Vectorizing text")
	# Make bag of words representation with CountVectorizer
	ings_list = [' '.join(x) for x in ingredients]
	test_list = [' '.join(x) for x in test_ingredients]

	vect = CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,2),
		max_features = 5000)
	# vect = Pipeline([
	# 	('vect', CountVectorizer()),
	# ])  
	bag_of_ingredients = vect.fit(ings_list)
	bag_of_ingredients = vect.transform(ings_list).tocsr()#.toarray()

	bag_of_test = vect.transform(test_list).tocsr()#.toarray()

	vectclasses = CountVectorizer(vocabulary=unique_cuisines_vocab)
	bag_of_classes = vectclasses.fit(classes)
	bag_of_classes = vectclasses.transform(classes).tocsr()#.toarray()

	unique_ingredients = set(item for sublist in ingredients for item in sublist)
	unique_cuisines = set(classes)

	print("Importing Data")
	# X, y, unique_cuisines = getdata(dataset='Train') # import the data
	# fil = 'one.hot.training.ingredients.sparse.npz'
	# fil2 = 'one.hot.training.classes.sparse.npz'
	# fil3 = 'one.hot.testing.ingredients.sparse.npz'
	# X = np.load(fil)['arr_0']

	# y = np.load(fil2)['arr_0']
	with open('foodtrain/train.json') as json_data:
		data = js.load(json_data)
		json_data.close()

	print("Data loaded")
	classes = [str(item['cuisine']) for item in data]

	X = bag_of_ingredients
	# Xtest = np.load(fil3)['arr_0']
	# # Split into training and validation sets
	rs = 19683
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, classes, \
			test_size=0.4, random_state=rs)

	cfr = svm.SVC(C=10,gamma=0.01,kernel='rbf')
	cfr.fit(X_train,y_train)
	predictions = cfr.predict(X_test)

	vectclasses = CountVectorizer(vocabulary=unique_cuisines_vocab)
	bag_of_classes = vectclasses.fit(predictions)
	bag_of_classes = vectclasses.transform(predictions).toarray()

	writestackgen(bag_of_classes,'StackGen.SVM.1-2grams.test.csv')
	# print("Training classifier")
	# # Train the classifier and fit to training data
	# # Grid search for RBF
	# Cs = np.logspace(1, 5, 10)
	# gammas = np.logspace(-4,-2,10)
	# classifier = GridSearchCV(estimator=svm.SVC(), \
	# 	param_grid=dict(C=Cs,gamma=gammas,kernel=['rbf']),
	# 	verbose=3,
	# 	n_jobs=-1,scoring='accuracy' )

	# classifier.fit(X_train, y_train)
	# print classifier.best_score_
	# print classifier.best_estimator_

	# Cs = np.logspace(0, 2, 6)
	# gammas = np.logspace(-2,2,5)
	# classifier = GridSearchCV(estimator=svm.SVC(), \
	# 	param_grid=dict(C=Cs,gamma=gammas,kernel=['rbf']),
	# 	verbose=3,
	# 	n_jobs=-1,scoring='accuracy' )

	# classifier.fit(X_train, y_train)
	# print classifier.best_score_
	# print classifier.best_estimator_

	# print("Making predictions on validation set")
	# # Make predictions on validation data
	# predictions = classifier.predict(X_test, batch_size=100, verbose=1)

	# # Take max value in preds rows as classification
	# pred = np.zeros((len(X_test)))
	# yint = np.zeros((len(X_test)))
	# for row in np.arange(0,len(predictions)) :
	# 	pred[row] = np.argmax(predictions[row])
	# 	yint[row] = np.argmax(y_test[row])

	# print("Classifier Accuracy = %d"%(metrics.accuracy_score(yint,pred)))

	#####
	# now to test
	#####
	# print("Testing classifier on Test data")
	# print("Re-train with full training set")
	# X, y, unique_cuisines = getdata(dataset='Train') # import the data

	# clf2 = nnk(X,unique_cuisines,lr=0.1)
	# f = clf2.fit(X, y, nb_epoch=35, batch_size=1000, 
	# 	validation_split=0.15, show_accuracy=True)

	# # print("Make predictions on test set")
	# # predictions = clf2.predict(Xtest, batch_size=25, verbose=1)
	# Xtest = np.genfromtxt('one.hot.testing.ingredients.csv',
	# 						delimiter = ',')

	# predictions = clf2.predict(Xtest, batch_size=100, verbose=1)
	# # Take max value in preds rows as classification
	# pred = np.zeros((len(Xtest)))
	# for row in np.arange(0,len(predictions)) :
	# 	pred[row] = np.argmax(predictions[row])

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
	unique_cuisines = sorted(list(unique_cuisines))
	newcuisines = []
	for row in np.arange(0,20) :
		newcuisines.append(unique_cuisines[row])

	predstr = []
	for row in np.arange(0,len(predictions)) :
		predstr.append(newcuisines[int(pred[row])])

	test_indices = np.genfromtxt('testing.indices.csv',
						delimiter = ',')
	# print("Storing predictions")
	writetest(test_indices,predstr,'SVM.c10.gamma01.5000feats.1-2grams.csv')