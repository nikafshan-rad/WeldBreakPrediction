import os
import sys
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from deap import base, creator
from deap import tools
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
import time
warnings.filterwarnings("ignore")
start_time = time.time()
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

print('Read data has been done.')
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
Xtrain = pd.DataFrame(Xtrain)
Xtest = pd.DataFrame(Xtest)
Ytrain = pd.DataFrame(Ytrain)
Ytest = pd.DataFrame(Ytest)
########################################################################
########################################################################
########################################################################
#######################################################################################
#######################################################################################
#######Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

number_of_gens = Xtrain.shape[1]

toolbox = base.Toolbox()

INT_MIN, INT_MAX = 0, 1
toolbox.register("attr_bool", random.randint, INT_MIN, INT_MAX )
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n = number_of_gens)
toolbox.register("population", tools.initRepeat, list , toolbox.individual)



def eval(individual):
    error_rate = np.inf
    new_individual = [] 
    for i in range(0,len(individual)-1):
        if individual[i] == 1:
            new_individual.append(i)
    
    if len(new_individual) > 0:
        new_X_train = Xtrain.ix[:,new_individual] 
        new_Y_train = Ytrain

        new_X_test = Xtest.ix[:,new_individual]
        new_Y_test = Ytest
        
        clf = GaussianNB()
        clf.fit(new_X_train,new_Y_train)
        
        score = str(classification_report(new_Y_test, clf.predict(new_X_test))).split('\n\n')[1].strip().split(' ')[7]
        
    return (float(score)),

toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxTwoPoint)# Two-point cross over
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)# Bit mutation 
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    pop = toolbox.population(n=30)
    
    CXPB, MUTPB, NGEN = 0.5, 0.2, 10
    
    print("Start of evolution")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        offspring = toolbox.select(pop, len(pop))
        
        offspring = list(map(toolbox.clone, offspring))
    
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print ('Total Time is  ' + str(time.time()-start_time) + ' seconds.')
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    
    new_features = []
    for i in range(0,len(best_ind)-1):
        if best_ind[i] == 1:
            new_features.append(i)
    ##############################################################
    ##############################################################
    ##############################################################
    ####Ensemble Learning
    print('#####################################################')
    new_X_train = Xtrain[new_features] 
    new_Y_train = Ytrain

    new_X_test = Xtest[new_features]
    new_Y_test = Ytest

    clf = GaussianNB()
    y_pred = clf.fit(new_X_train, new_Y_train).predict(new_X_test)
    f = open('GNB_GA.txt','w')
    print(classification_report(Ytest,y_pred))
    f.write(classification_report(Ytest,y_pred))
    f.write('\n')
    f.write('ROC = ' + str(roc_auc_score(Ytest,y_pred)))
    f.write('\n')
    f.close()

if  __name__ =='__main__':main()
