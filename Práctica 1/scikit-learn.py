from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn import preprocessing

from abc import ABCMeta,abstractmethod
import numpy as np
from Datos import Datos

dataset = Datos('ConjuntoDatos/german.csv')
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
print(">>>---- GAUSSIAN german.csv")
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print("\n\n>>>--- MULTINOMIAL german.csv sin correcion de laplace..... ")

clf = MultinomialNB(alpha = 0,fit_prior= True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")
print(">>> MULTINOMIAL german.csv con correcion de laplace..... ")
clf = MultinomialNB(alpha = 1,fit_prior= True)
y_pred = clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")


print(">>> CATEGORICAL german.csv sin correcion de laplace..... ")
x = dataset.datos.iloc[:,:-1]
y = dataset.datos.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
ctl = CategoricalNB(fit_prior= True,alpha = 0)
y_pred = ctl.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")

print(">>> CATEGORICAL german.csv con correcion de laplace..... ")

ctl = CategoricalNB(fit_prior = True, alpha = 1)
y_pred = ctl.fit(X_train,y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
porcentaje = (y_test != y_pred).sum() / X_test.shape[0] * 100
print("Porcentaje error: " + str(porcentaje) + "%")


