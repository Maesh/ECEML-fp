import csv as csv
import scipy as scipy
import numpy as np

# Set random state before keras imports
rs = 19683
from sklearn import svm, cross_validation, preprocessing, metrics
from sklearn.grid_search import GridSearchCV
from collections import Counter

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def writetest(idx,Xpreds, fil='ensem1.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([int(idx[row]),Xpreds[row]])

if __name__ == '__main__':
	res1 = np.genfromtxt('DNN.512.256.64.PReLU.csv',delimiter=',',
		skip_header=1,dtype=str)
	res2 = np.genfromtxt('NN.512.256.64.csv',delimiter=',',
		skip_header=1,dtype=str)
	res3 = np.genfromtxt('SVM.c10.gamma01.5000feats.1-2grams.csv',
		delimiter=',',skip_header=1,dtype=str)
	res4 = np.genfromtxt('NN.512.256.64.PReLU.5000feats.csv',
		delimiter=',',skip_header=1,dtype=str)


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

	# Convert strings to ints for comparison
	res1ints = []
	res2ints = []
	res3ints = []
	res4ints = []
	for row in np.arange(0,len(res1)) :
		res1ints.append(unique_cuisines_vocab[res1[row,1]])
		res2ints.append(unique_cuisines_vocab[res2[row,1]])
		res3ints.append(unique_cuisines_vocab[res3[row,1]])
		res4ints.append(unique_cuisines_vocab[res4[row,1]])

	# Compute correlation coefficient
	mats = [res1ints,res2ints,res3ints,res4ints]
	cc = np.corrcoef(mats)

	#  ==> no cross correlation above 0.8, nice

	# take the vote
	# for each row, look at the votes. Since res2 had the highest score
	# in the event of tie or total disagreement, give it to res2
	matsarray = np.array(mats).T # (samples x methods)
	ensemb = []
	for row in np.arange(0,matsarray.shape[0]) :
		ctr = Counter(matsarray[row]) # find counts of each thing
		# Returns class with highest vote, or if three way tie, returns
		# first vote it received. Carefully place best metric in first
		# slot
		ensemb.append(ctr.most_common()[0][0])

	# ensemb contains the classes in int form, convert back to strings
	# Unpleasant strategy but it works
	ensembStrings = []
	for row in np.arange(0,len(ensemb)) :
		for key, val in unique_cuisines_vocab.items() :
			if val == ensemb[row] :
				ensembStrings.append(key)

	test_indices = np.genfromtxt('testing.indices.csv',
						delimiter = ',')
	writetest(test_indices,ensembStrings,fil='ensem.NN.DNN.SVM.DNN5000.csv')