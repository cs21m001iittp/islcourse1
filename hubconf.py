
#endsem
# Importing libraries
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
from sklearn import preprocessing
from sklearn import model_selection
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

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = None
  lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    # 'C': [1.0, 2.0],
    'solver': ['liblinear'],

  }
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = None
  rf_param_grid = {
    'n_estimators': list(range(10,101,10)),
    'max_features': list(range(6,32,5)),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 5, 10, None]
  }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc_ovo', 'roc_auc_ovr']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  grid_search_cv = None
  grid_search_cv = model_selection.GridSearchCV(model1, param_grid, cv=cv, scoring=metrics, refit=False).fit(X, y)
  cv_results = grid_search_cv.cv_results_
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation

  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  top1_scores = []

  for metric in metrics:
    i = list(cv_results[f'rank_test_{metric}']).index(1)
    top_score = list(cv_results[f'mean_test_{metric}'])[i]
    top1_scores.append(top_score)
  
  return top1_scores



#####Part3######
class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.num_classes = num_classes

    self.fc_encoder = nn.Linear(inp_dim, hid_dim) # write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(hid_dim, inp_dim) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(hid_dim, num_classes) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax = nn.Softmax(dim=1) #write your code - softmax object
    
  def forward(self,x):
    x = nn.Flatten()(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    yground_1h = nn.functional.one_hot(yground, num_classes=self.num_classes)

    # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    lc1 = -torch.mean(yground_1h*torch.log(y_pred+0.00001))
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X, y = None, None
  X, y = get_data_mnist()
  X, y = torch.tensor(X), torch.tensor(y)
  # write your code
  return X,y

def get_loss_on_single_point(mynn, x0, y0):
  X0 = torch.unsqueeze(x0, dim=0)   
  Y0 = torch.unsqueeze(y0, dim=0)

  y_pred, xencdec = mynn(X0)
  lossval = mynn.loss_fn(X0,Y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn, X, y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimizer.step()
    print(f"Epoch {i+1}: ", lval.item())
    
  return mynn
