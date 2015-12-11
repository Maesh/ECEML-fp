from foodio import *
from scipy import sparse
from sklearn import cross_validation, preprocessing, metrics, svm
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import numpy as np

if __name__ == '__main__':
	
	X, y, Xtest, unique_cuisines,indices = getdata() # import the data

	# sparsify because why not
	Xsparse = sparse.
	# Split into training and validation sets
	rs = 19683
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.4, random_state=rs)

	# Set up cross validation
	# cv = cross_validation.StratifiedKFold(y, 5,shuffle=True,\
	# 	random_state=rs)

	# Try Normalization
	# Xnorm = preprocessing.normalize(X).copy()

	# Gridsearch for best parameters
	Cs = np.logspace(-2, 3, 6)
	gammas = np.logspace(-3,3,7)
	classifier = GridSearchCV(estimator=svm.SVC(), \
		param_grid=dict(C=Cs,gamma=gammas,kernel=['rbf']) )

	classifier.fit(X, y)