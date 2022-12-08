#endsem
# Importing libraries
import torch
from torch import nn
import torch.optim as optim

import sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles, load_digits
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from torchvision.transforms import ToTensor
import torch.nn.functional as Fun
# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

device = "cuda" if torch.cuda.is_available() else "cpu"
###### PART 1 #######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples = n_points)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  pass
  X, y = make_circles(n_samples = n_points)
  return X,y

def get_data_mnist():
  digits= load_digits()
  X = digits.data
  y = digits.target
  return X,y

def build_kmeans(X=None,k=10):
  pass
  km = KMeans(n_clusters=k).fit(X)
  return km

def assign_kmeans(km=None,X=None):
  pass
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v

#####part2########
def build_lr_model(X=None, y=None):
  # write your code...
  # Build logistic regression, refer to sklearn
  scaler = preprocessing.StandardScaler().fit(X)
  X_train = scaler.transform(X)
  lr_model = None
  lr_model = LogisticRegression(random_state=0).fit(X_train, y)
  return lr_model

def build_rf_model(X=None, y=None):
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model = None
  rf_model = RandomForestClassifier(random_state=0, max_depth=5).fit(X, y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  ypred = model1.predict(X)

  acc = metrics.accuracy_score(y, ypred)
  prec = metrics.precision_score(y, ypred, average='macro')
  rec = metrics.recall_score(y, ypred, average='macro')
  f1 = metrics.f1_score(y, ypred, average='macro')
  fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=2)
  auc = metrics.auc(fpr, tpr)
  return acc, prec, rec, f1, auc

