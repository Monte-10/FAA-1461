

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def getErroresLogisticRegresion():
    dataset1 = pd.read_csv('ConjuntosDatosP2/pima-indians-diabetes.csv')
    X = dataset1[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    y = dataset1[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    logreg = LogisticRegression(random_state=16, max_iter=10)
    logreg.fit(X_train, y_train)
    score = logreg.score(X_test, y_test)

    dataset2 = pd.read_csv('ConjuntosDatosP2/wdbc.csv')
    X2 = dataset2[['Atributo1','Atributo2','Atributo3','Atributo4','Atributo5','Atributo6','Atributo7','Atributo8','Atributo9','Atributo10','Atributo11','Atributo12','Atributo13','Atributo14','Atributo15'
              ,'Atributo16','Atributo17','Atributo18','Atributo19','Atributo20','Atributo21','Atributo22','Atributo23','Atributo24','Atributo25','Atributo26','Atributo27','Atributo28','Atributo29','Atributo30']].values
    y2 = dataset2[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=16)
    clf2 = LogisticRegression(random_state=16, solver='lbfgs', max_iter=10).fit(X2, y2)
    clf2.fit(X_train, y_train)
    score2 = clf2.score(X_test, y_test)
    
    print(f'El error para pima es: {1-score}')
    print(f'El error para wdbc es: {1- score2}')


def getErroresSGDClassifier(normaliza):
    dataset1 = pd.read_csv('ConjuntosDatosP2/pima-indians-diabetes.csv')
   
    X = dataset1[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    y = dataset1[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    #It is particularly important to scale the features when using the SGD Classifier.
    if normaliza is True:# print("normalizing data")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
    
    clf = SGDClassifier(loss="log",eta0=1.0)
    clf.fit(X_train,  y_train)
    y_pred = clf.predict(X_test)

    print('Error for pima: {:.2f}'.format(1-accuracy_score(y_test, y_pred)))



    dataset2 = pd.read_csv('ConjuntosDatosP2/wdbc.csv')
   
    X2 = dataset2[['Atributo1','Atributo2','Atributo3','Atributo4','Atributo5','Atributo6','Atributo7','Atributo8','Atributo9','Atributo10','Atributo11','Atributo12','Atributo13','Atributo14','Atributo15'
              ,'Atributo16','Atributo17','Atributo18','Atributo19','Atributo20','Atributo21','Atributo22','Atributo23','Atributo24','Atributo25','Atributo26','Atributo27','Atributo28','Atributo29','Atributo30']].values
    y2 = dataset2[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=16)    #It is particularly important to scale the features when using the SGD Classifier.
    if normaliza is True:# print("normalizing data")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
    '''as already mentioned above SGD-Classifier is a Linear classifier with SGD training. Which linear classifier is used is determined with the hypter parameter loss. So, if I write clf 
    = SGDClassifier(loss=‘hinge’) it is an implementation of Linear SVM and if I write clf = SGDClassifier(loss=‘log’) it is an implementation of Logisitic regression.'''    
    clf = SGDClassifier(loss="log",eta0=1.0)
    clf.fit(X_train,  y_train)
    y_pred = clf.predict(X_test)

    print('Error for wdbc: {:.2f}'.format(1- accuracy_score(y_test, y_pred)))
def main():
    getErroresLogisticRegresion()
    getErroresSGDClassifier(normaliza = False)
if __name__ == "__main__":
    main()