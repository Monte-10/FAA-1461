
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import math
import pandas as pd

from abc import ABCMeta,abstractmethod
import numpy as np
from Datos import Datos

dataset = Datos('ConjuntoDatos/german.csv')
dataset2 = Datos('ConjuntoDatos/tic-tac-toe.csv')
datasetOHE = pd.read_csv('ConjuntoDatos/tic-tac-toe.csv')

atrs = preprocessing.OneHotEncoder()


x = atrs.fit_transform(dataset.datos.iloc[:,:-1]).toarray()

#print(x)
y = dataset.datos.iloc[:,-1]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

print('\n\n>>>>>>>>>>>GERMAN<<<<<<<<<<<<<<')
print("\n\n>>> CON TRAIN_TEST_SPLIT..... ")

print("\n>>>---- GAUSSIAN german.csv")
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print("\n\n>>>--- MULTINOMIAL german.csv sin correcion de laplace..... ")

clf = MultinomialNB(alpha=1.0*math.e**10,fit_prior = True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print("\n\n>>> MULTINOMIAL german.csv con correcion de laplace..... ")
clf = MultinomialNB(alpha = 1,fit_prior = True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")


print("\n\n>>> CON CROSS_VAL_SCORE..... ")
x = atrs.fit_transform(dataset.datos.iloc[:,:-1]).toarray()
y = dataset.datos.iloc[:,-1]
print("\n\n>>>---- GAUSSIAN german.csv")
gnb = GaussianNB()
print(cross_val_score(gnb, x, y, cv=3))

print("\n\n>>>--- MULTINOMIAL german.csv sin correcion de laplace..... ")
clf = MultinomialNB(alpha=1.0*math.e**10,fit_prior= True)
print(cross_val_score(clf, x, y, cv=3))

print("\n\n>>> MULTINOMIAL german.csv con correcion de laplace..... ")
clf = MultinomialNB(alpha = 5,fit_prior= True)
print(cross_val_score(clf, x, y, cv=3))

'''TIC TAC TOE'''
print('\n\n\n\n\n\n>>>>>>>>>>>TIC TAC TOE<<<<<<<<<<<<<<')
x = atrs.fit_transform(dataset2.datos.iloc[:,:-1]).toarray()
y = dataset2.datos.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print("\n>>> CON TRAIN_TEST_SPLIT..... ")

print("\n\n>>>---- GAUSSIAN tic-tac-toe.csv")
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print("\n\n>>>--- MULTINOMIAL tic-tac-toe.csv sin correcion de laplace..... ")

clf = MultinomialNB(alpha=1.0*math.e**10,fit_prior = True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print("\n\n>>> MULTINOMIAL tic-tac-toe.csv con correcion de laplace..... ")
clf = MultinomialNB(alpha = 1,fit_prior = True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")


print("\n\n>>> CON VALIDACION CRUZADA..... ")
x = atrs.fit_transform(dataset2.datos.iloc[:,:-1]).toarray()
y = dataset2.datos.iloc[:,-1]
print(">>>---- GAUSSIAN tic-tac-toe.csv")
gnb = GaussianNB()
print(cross_val_score(gnb, x, y, cv=3))

print("\n\n>>>--- MULTINOMIAL tic-tac-toe.csv sin correcion de laplace..... ")
clf = MultinomialNB(alpha=1.0*math.e**10,fit_prior= True)
print(cross_val_score(clf, x, y, cv=3))

print("\n\n>>> MULTINOMIAL tic-tac-toe.csv con correcion de laplace..... ")
clf = MultinomialNB(alpha = 5,fit_prior= True)
print(cross_val_score(clf, x, y, cv=3))

print('\n\n\n\n\n\n>>>>>>>>>>> TIC TAC TOE -> ONE HOT ENCODER (Validación Simple) <<<<<<<<<<<<<<')

atrs = list(datasetOHE.columns)
atrs.remove('Class')

ohe = OneHotEncoder()

x = datasetOHE[atrs].values
y = datasetOHE['Class'].values
x = ohe.fit_transform(x).toarray()

err_gnb = []
err_clf = []

gnb = GaussianNB()
clf = MultinomialNB()

for i in range(0,3):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    
    gnb.fit(X_train,Y_train)
    err_gnb.append(1-gnb.score(X_test,Y_test))
    
    clf.fit(X_train, Y_train)
    err_clf.append(1-clf.score(X_test,Y_test))
    
print("\n\n\n>>>---- ERROR GAUSSIAN OHE tic-tac-toe.csv:\n\n")
for item in err_gnb:
    print(item)
print("\n\n\n>>>---- ERROR MULTINOMIAL OHE tic-tac-toe.csv: \n\n")
for item in err_clf:
    print(item)
