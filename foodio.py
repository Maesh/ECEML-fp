"""
Function storage for getting and processing data, as well as 
writing to csv.
"""
import numpy as np
import json as js
import csv
import string
import nltk

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
nltk.data.path.append('/media/maesh/Charming/nltk_data')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def writetest(idx,Xpreds, fil='NN.512.256.64.csv') :
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([idx[row],Xpreds[row]])

def writedata() :
	"""
	Changing this to a function for actually tokenizing and 
	writing data to csv
	"""
	

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

	vect = CountVectorizer(tokenizer=LemmaTokenizer())
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

	print("Writing files")
	# Now to actually write the data
	fil = 'one.hot.training.ingredients.sparse.csv'
	fil2 = 'one.hot.training.classes.sparse.csv'
	fil3 = 'one.hot.testing.ingredients.sparse.csv'
	fil4 = 'testing.indices.sparse.csv'

	np.savez(fil, bag_of_ingredients)
	np.savez(fil2, bag_of_classes)
	np.savez(fil3, bag_of_test)
	# csv.field_size_limit(1000000000)
	# outwriter = csv.writer(open(fil,'w'),delimiter=",")
	# # rows = np.arange(0,len(bag_of_ingredients))
	# rows = np.arange(0,bag_of_ingredients.shape[0])
	# for row in rows :
	# 	outwriter.writerow(bag_of_ingredients[row])

	# csv.field_size_limit(1000000000)
	# outwriter = csv.writer(open(fil2,'w'),delimiter=",")
	# # rows = np.arange(0,len(bag_of_classes))
	# rows = np.arange(0,bag_of_classes.shape[0])
	# for row in rows :
	# 	outwriter.writerow(bag_of_classes[row])

	# csv.field_size_limit(1000000000)
	# outwriter = csv.writer(open(fil3,'w'),delimiter=",")
	# # rows = np.arange(0,len(bag_of_test))
	# rows = np.arange(0,bag_of_test.shape[0])
	# for row in rows :
	# 	outwriter.writerow(bag_of_test[row])

	# csv.field_size_limit(1000000000)
	# outwriter = csv.writer(open(fil4,'w'),delimiter=",")
	# rows = np.arange(0,len(test_indices))
	# for row in rows :
	# 	outwriter.writerow([test_indices[row]])


def getdata(write=False,dataset='Train') :
	"""
	Gets all the data in. 
	"""
	if dataset == 'Train' :
		print("Get training ingredients")
		ingredients = np.genfromtxt('one.hot.training.ingredients.ng1-2.csv',
						delimiter = ',')

		print("Get training classes")
		classes = np.genfromtxt('one.hot.training.classes.ng1-2.csv',
						delimiter = ',')
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
		print "Number of recipes = %d"%(len(ingredients))
		print "Number of unique ingredients = %d"%(len(ingredients[0]))
		print "Number of unique cuisines = %d"%(len(unique_cuisines))
		return ingredients,classes,unique_cuisines

	if dataset == 'Test' :
		print("Get testing ingredients")
		test_ingredients = np.genfromtxt('one.hot.testing.ingredients.csv',
						delimiter = ',')

		print("Get testing indices")
		indices = np.genfromtxt('testing.indices.csv',
						delimiter = ',')
		return test_ingredients, indices
	#unique_ingredients = set(item for sublist in ingredients for item in sublist)
	# Hack but it makes it easier

def getsparsedata() :
	"""
	Produces sparse data version for SVM
	"""
	import numpy as np
	import json as js
	from scipy.sparse import csr_matrix

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