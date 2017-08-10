# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Gmm:

	def __init__(self):
		self.cat_num = 3

	def set_features(self):
		iris = load_iris()
		features = iris.data
		#labels = iris.target
		#return features,labels
		return features

	def gmm(self):
		#X,_ = self.set_features()
		X = self.set_features()
		clf = GaussianMixture(n_components = self.cat_num)
		clf.fit(X)
		pred = clf.predict(X)
		np.save("pred.npy",pred)
		self.show()

	def show(self):
		X,Y = self.set_features()
		pca = PCA(n_components = 2)
		pca.fit(X)
		X = pca.fit_transform(X)
		pred = np.load("pred.npy")


		for number,p in enumerate(pred):
			if p == 0:
				plt.scatter(X[number][0],X[number][1],color = "red")
			elif p == 1:
				plt.scatter(X[number][0],X[number][1],color = "blue")
			elif p == 2:
				plt.scatter(X[number][0],X[number][1],color = "green")

		plt.show()

if __name__ == "__main__":
	g = Gmm()
	g.gmm()
	