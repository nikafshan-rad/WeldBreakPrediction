import os
import sys
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Imputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.svm import SVR
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.random_projection import check_random_state
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")
########################################################################
ok_features = dict()

path = "..\\..\\OK"
for filename in os.listdir(path):
    f = open(path+'\\'+filename,'r')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    temp = ""
    tokens = lines[3].split(' ')
    tokens = tokens[3].split('\t')
    id = tokens[1]
    #################################
    for i in range(4,6):
        tokens = lines[i].split(' ')
        tokens = tokens[3].split('\t')
        temp += str(tokens[1]) + ' '
    #################################
    tokens = lines[6].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    #################################
    for i in range(8,10):
        tokens = lines[i].split(' ')
        tokens = tokens[3].split('\t')
        temp += str(tokens[1]) + ' '
    #################################
    tokens = lines[10].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    #################################
    for i in range(11,17):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' ' 
    #################################  
    for i in range(17,19):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' ' 
    #################################
    tokens = lines[19].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[20].split(' ')
    tokens = tokens[1].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[21].split(' ')
    tokens = tokens[1].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[22].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    for i in range(23,24):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' '    
    ################################
    tokens = lines[24].split(' ')
    tokens = tokens[1].split('\t')
    temp += str(tokens[1]) + ' '
    
    temp += str(len(lines)-26)
    
    ok_features[id] = temp
        
    for i in range(26,len(lines)):
        tokens = lines[i].split(' ')
        
#############################################################
#############################################################
#############################################################
wb_features = dict()

path = "..\\..\\WB"
for filename in os.listdir(path):
    f = open(path+'\\'+filename,'r')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    temp = ""
    tokens = lines[3].split(' ')
    tokens = tokens[3].split('\t')
    id = tokens[1]
    #################################
    for i in range(4,6):
        tokens = lines[i].split(' ')
        tokens = tokens[3].split('\t')
        temp += str(tokens[1]) + ' '
    #################################
    tokens = lines[6].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    #################################
    for i in range(8,10):
        tokens = lines[i].split(' ')
        tokens = tokens[3].split('\t')
        temp += str(tokens[1]) + ' '
    #################################
    tokens = lines[10].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    #################################
    for i in range(11,17):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' ' 
    #################################  
    for i in range(17,19):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' ' 
    #################################
    tokens = lines[19].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[20].split(' ')
    tokens = tokens[1].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[21].split(' ')
    tokens = tokens[1].split('\t')
    temp += tokens[1] + ' '
    ################################
    tokens = lines[22].split(' ')
    tokens = tokens[2].split('\t')
    temp += tokens[1] + ' '
    for i in range(23,24):
        tokens = lines[i].split(' ')
        tokens = tokens[1].split('\t')
        temp += str(tokens[1]) + ' '    
    ################################
    tokens = lines[24].split(' ')
    tokens = tokens[1].split('\t')
    temp += str(tokens[1]) + ' '    
    
    temp += str(len(lines)-26)
    wb_features[id] = temp
            
    for i in range(26,len(lines)):
        tokens = lines[i].split(' ')     

#######################################################################
#######################################################################
#######################################################################
data = pd.DataFrame([str(line).split(' ') for line in ok_features.values()])
data.columns = [['Lasparameter.Ingang-Dikte','Lasparameter.Ingang-Breedte','Lasparameter.Ingang-Materiaal','Lasparameter.Uitgang-Dikte','Lasparameter.Uitgang-Breedte','Lasparameter.Uitgang-Materiaal','Lasparameter.P2','Lasparameter.P3','Lasparameter.P4','Lasparameter.P5','Lasparameter.P6','Lasparameter.P7','Lasparameter.Vonksnelheid','Lasparameter.Lassnelheid','Lasparameter.StroomReferentie','Lasparameter.Stroomtoename','Lasparameter.Stuikstroomtijd','Lasparameter.AfkoeltijdStuik','Lasparameter.Stuikdruk','Lasparameter.Stuiksnelheid','count']]
data['label'] = [1 for x in np.ones(len(data))]

data1 = pd.DataFrame([str(line).split(' ') for line in wb_features.values()])
data1.columns = [['Lasparameter.Ingang-Dikte','Lasparameter.Ingang-Breedte','Lasparameter.Ingang-Materiaal','Lasparameter.Uitgang-Dikte','Lasparameter.Uitgang-Breedte','Lasparameter.Uitgang-Materiaal','Lasparameter.P2','Lasparameter.P3','Lasparameter.P4','Lasparameter.P5','Lasparameter.P6','Lasparameter.P7','Lasparameter.Vonksnelheid','Lasparameter.Lassnelheid','Lasparameter.StroomReferentie','Lasparameter.Stroomtoename','Lasparameter.Stuikstroomtijd','Lasparameter.AfkoeltijdStuik','Lasparameter.Stuikdruk','Lasparameter.Stuiksnelheid','count']]
data1['label'] = [0 for x in np.ones(len(data1))]

data = pd.concat([data, data1])
for col in data.columns.values:
    data[col] = data[col].astype(float)

########################################################################
########################################################################
########################################################################
data_X = data [['Lasparameter.Ingang-Dikte','Lasparameter.Ingang-Breedte','Lasparameter.Ingang-Materiaal','Lasparameter.Uitgang-Dikte','Lasparameter.Uitgang-Breedte','Lasparameter.Uitgang-Materiaal','Lasparameter.P2','Lasparameter.P3','Lasparameter.P4','Lasparameter.P5','Lasparameter.P6','Lasparameter.P7','Lasparameter.Vonksnelheid','Lasparameter.Lassnelheid','Lasparameter.StroomReferentie','Lasparameter.Stroomtoename','Lasparameter.Stuikstroomtijd','Lasparameter.AfkoeltijdStuik','Lasparameter.Stuikdruk','Lasparameter.Stuiksnelheid','count']]
data_Y = data [['label']]


for col in data_X.columns.values:
    if data_X[col].dtypes=='object':
        print(str(col))
        ll = list(set(data_X))
        for i in range(0,len(ll)):
            data_X[col+str(i)] = data_X[col].map(lambda x:1 if x==ll[i] else 0)
        data_X.pop(col)
                  
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data_X = imp.fit_transform(data_X)
scaler = StandardScaler()
scaler.fit(data_X)
data_X = scaler.transform(data_X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)
########################################################################
########################################################################
########################################################################
params = {'n_estimators': 10, 'max_depth': 3, 'subsample': 0.5,
                  'learning_rate': 0.89, 'min_samples_leaf': 1, 'random_state': 5}

clfs = [
        ensemble.GradientBoostingClassifier(**params),
        BernoulliNB(),
        DecisionTreeClassifier(random_state=0),
        svm.SVC(kernel='rbf', probability=True),
        SGDClassifier(loss="modified_huber",penalty='l1'),
        RandomForestClassifier(n_estimators=9),
        ensemble.AdaBoostClassifier(),
        svm.SVC(kernel='linear', probability=True),
        MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(150,50,15,5,3), random_state=1),
        neighbors.KNeighborsClassifier(n_neighbors=5),
        NearestCentroid(metric='euclidean', shrink_threshold=None),
        GaussianNB()
        
        
       ]
clfs_name = ['GradientBoostingClassifier', 'Bernoulli Naive Bayes',
             'DecisionTreeClassifier', 'SVM (rbf)', 'Stochastic Gradient Descent',
             'Radom Forest Classifier', 'Ada Boost Classifier',
             'SVM (linear)' , 'Multi Layer Perceptron', 'K Nearest Neighbors',
             'Nearest Centroid Classifier', 'Guassian Naive Bayes'
            ]

f = open("models.txt","w")
for i in range(0,len(clfs)):
    y_pred = clfs[i].fit(Xtrain, Ytrain).predict(Xtest)    
    f.write("########################################" +
            clfs_name[i] + 
            "########################################" + '\n')
    f.write(classification_report(Ytest,y_pred))
    f.write('\n')
    f.write('ROC = ' + str(roc_auc_score(Ytest,y_pred)))
    f.write('\n')
    
    print(str(i+1) + " from " + str(len(clfs)) + " has been done.") 
	
f.close()


    




