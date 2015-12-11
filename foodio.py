"""
Function storage for getting and processing data, as well as 
writing to csv.
"""

def writetest(idx,Xpreds, fil='NN.512.256.64.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([idx[row],Xpreds[row]])

def getdata() :
	"""
	Gets all the data in. 
	"""
	import numpy as np
	import json as js

	#Training
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

	X_train = np.zeros((len(ingredients), len(unique_ingredients)))

	# Compile feature matrix. Could be sparse but dense is fine for us.
	# Each feature is an ingredient. Each row is a recipe. For each 
	# column (feature), a 1 indicates the recipe has that ingredient,
	# while a 0 indicates it does not.
	for d,dish in enumerate(ingredients):
		for i,ingredient in enumerate(unique_ingredients):
			if ingredient in dish:
				X_train[d,i] = 1

	# Also need to ensure
	y_train=np.zeros((len(classes),len(unique_cuisines)))
	for c,clas in enumerate(classes):
		for q,cuisine in enumerate(unique_cuisines):
			if cuisine in clas:
				y_train[c,q] = 1

	# Testing
	# to test:
	with open('foodtest/test.json') as json_data:
		data = js.load(json_data)
		json_data.close()

	ingredients = [item['ingredients'] for item in data]
	indices = [item['id'] for item in data]

	# print len(data)
	# print (classes)
	print "Number of recipes = %d"%(len(ingredients))
	print "Number of unique ingredients = %d"%(len(unique_ingredients))

	Xtest = np.zeros((len(ingredients), len(unique_ingredients)))

	# Compile feature matrix. Could be sparse but dense is fine for us.
	# Each feature is an ingredient. Each row is a recipe. For each 
	# column (feature), a 1 indicates the recipe has that ingredient,
	# while a 0 indicates it does not.
	for d,dish in enumerate(ingredients):
		for i,ingredient in enumerate(unique_ingredients):
			if ingredient in dish:
				Xtest[d,i] = 1

	print Xtest.shape
	return X_train,y_train,Xtest,unique_cuisines,indices

def getsparsedata() :
	"""
	Produces sparse data version for SVM
	"""
	import numpy as np
	import json as js

	#Training
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

	X_train = np.zeros((len(ingredients), len(unique_ingredients)))

	# Compile feature matrix. Could be sparse but dense is fine for us.
	# Each feature is an ingredient. Each row is a recipe. For each 
	# column (feature), a 1 indicates the recipe has that ingredient,
	# while a 0 indicates it does not.
	for d,dish in enumerate(ingredients):
		for i,ingredient in enumerate(unique_ingredients):
			if ingredient in dish:
				X_train[d,i] = 1

	# Also need to ensure
	y_train=np.zeros((len(classes),len(unique_cuisines)))
	for c,clas in enumerate(classes):
		for q,cuisine in enumerate(unique_cuisines):
			if cuisine in clas:
				y_train[c,q] = 1

	# Testing
	# to test:
	with open('foodtest/test.json') as json_data:
		data = js.load(json_data)
		json_data.close()

	ingredients = [item['ingredients'] for item in data]
	indices = [item['id'] for item in data]

	# print len(data)
	# print (classes)
	print "Number of recipes = %d"%(len(ingredients))
	print "Number of unique ingredients = %d"%(len(unique_ingredients))

	Xtest = np.zeros((len(ingredients), len(unique_ingredients)))

	# Compile feature matrix. Could be sparse but dense is fine for us.
	# Each feature is an ingredient. Each row is a recipe. For each 
	# column (feature), a 1 indicates the recipe has that ingredient,
	# while a 0 indicates it does not.
	for d,dish in enumerate(ingredients):
		for i,ingredient in enumerate(unique_ingredients):
			if ingredient in dish:
				Xtest[d,i] = 1

	print Xtest.shape
	return X_train,y_train,Xtest,unique_cuisines,indices