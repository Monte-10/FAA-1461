
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn import preprocessing
import math

from abc import ABCMeta,abstractmethod
import numpy as np
from Datos import Datos

dataset = Datos('Práctica 1/ConjuntoDatos/german.csv')
dataset2 = Datos('Práctica 1/ConjuntoDatos/german.csv')
#datos = dataset.datos[dataset.datos.keys()[0]:dataset.datos.keys()[-2]]
#clase = dataset.datos[dataset.datos.keys()[-1]]
#print(dataset.datos[0])


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
