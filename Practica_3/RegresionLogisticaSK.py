from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
def getErroresLogisticRegresion(normaliza,nombreFichero1,nombreFichero2,n_epocas,cte_aprendizaje):
    dataset1 = pd.read_csv(nombreFichero1)
    X = dataset1[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    y = dataset1[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    if normaliza is True:# print("normalizing data")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
    

    logreg = LogisticRegression(random_state=16,solver='lbfgs', max_iter=n_epocas, C = cte_aprendizaje)
    logreg.fit(X_train, y_train)
    score = logreg.score(X_test, y_test)

    dataset2 = pd.read_csv(nombreFichero2)
    X2 = dataset2[['Atributo1','Atributo2','Atributo3','Atributo4','Atributo5','Atributo6','Atributo7','Atributo8','Atributo9','Atributo10','Atributo11','Atributo12','Atributo13','Atributo14','Atributo15'
              ,'Atributo16','Atributo17','Atributo18','Atributo19','Atributo20','Atributo21','Atributo22','Atributo23','Atributo24','Atributo25','Atributo26','Atributo27','Atributo28','Atributo29','Atributo30']].values
    y2 = dataset2[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=16)


    if normaliza is True:# print("normalizing data")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 

    clf2 = LogisticRegression(random_state=16, solver='lbfgs', max_iter=n_epocas, C = cte_aprendizaje).fit(X2, y2)
    clf2.fit(X_train, y_train)
    score2 = clf2.score(X_test, y_test)
    
    
    return score, score2


def getErroresSGDClassifier(normaliza,nombreFichero1,nombreFichero2,n_epocas,cte_aprendizaje):
    dataset1 = pd.read_csv(nombreFichero1)
   
    X = dataset1[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    y = dataset1[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    #It is particularly important to scale the features when using the SGD Classifier.
    if normaliza is True:# print("normalizing data")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
    
    clf = SGDClassifier(loss="log",eta0=cte_aprendizaje,max_iter = n_epocas)
    clf.fit(X_train,  y_train)
    # y_pred = clf.predict(X_test)
    error_1 = clf.score(X_test,y_test)
    # print('Error for pima: {:.2f}'.format(1-accuracy_score(y_test, y_pred)))



    dataset2 = pd.read_csv(nombreFichero2)
   
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
    clf = SGDClassifier(loss="log",eta0=cte_aprendizaje,max_iter = n_epocas)
    clf.fit(X_train,  y_train)
    # y_pred = clf.predict(X_test)
    error_2 = clf.score(X_test,y_test)
    # print('Error for wdbc: {:.2f}'.format(1- accuracy_score(y_test, y_pred)))
    return error_1,error_2 

def plotErrores(error1, error2,error3,error4,listaValores):
    plt.plot(listaValores, error1, label='SGDpima')
    plt.plot(listaValores, error2, label='SGDwdbc')
    plt.plot(listaValores, error3, label='RLpima')
    plt.plot(listaValores, error4, label='RLwdbc')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.xlabel('Constante de aprendizaje')
    plt.show()


def main():
    n_epocas = 100
    gradiente = 1.0
    score1_total = [] #pima RL
    score2_total = [] #wdbc RL
    score3_total = [] #pima SGD
    score4_total = [] #wdbc SGD
    valores = []
    valor = 1
    for i in range(5):
        valor = valor * 0.1
        valores.append(valor)
    print(valores)
    for i in valores:
        
        score1,score2 = getErroresLogisticRegresion(False,'ConjuntosDatosP2/pima-indians-diabetes.csv','ConjuntosDatosP2/wdbc.csv',n_epocas,i)
        error1,error2 = getErroresSGDClassifier(False,'ConjuntosDatosP2/pima-indians-diabetes.csv','ConjuntosDatosP2/wdbc.csv',n_epocas,i)
        score1_total.append(score1)
        score2_total.append(score2)
        score3_total.append(error1)
        score4_total.append(error2)
    plotErrores(score3_total,score4_total,score1_total,score2_total,valores)
if __name__ == "__main__":
    main()