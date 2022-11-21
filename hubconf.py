
#endsem

import torch
from torch import nn
import torch.optim as optim

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles, load_digits
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples = n_points)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples = n_points)
  # write your code ...
  return X,y

def get_data_mnist():
  
  # write your code here
  # Refer to sklearn data sets
  
  digits= load_digits()
  X = digits.data
  y = digits.target
  
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(n_clusters=k).fit(X)
  # write your code ...
  return km

def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v

###### PART 2 ######

