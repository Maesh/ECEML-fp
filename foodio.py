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

def getdata(ngram_range=(1,1)) :
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

	vect = CountVectorizer(tokenizer=LemmaTokenizer(),ngram_range=ngram_range,
		max_features=5000)
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

	return bag_of_ingredients, bag_of_classes, unique_cuisines,classes, test_indices, bag_of_test