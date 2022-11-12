

#TODO: diferencias entre la implementacion de sk y la nuestra
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split



def main():
    dataset1 = pd.read_csv('ConjuntosDatosP2/pima-indians-diabetes.csv')
    X = dataset1[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    y = dataset1[['Class']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg.fit(X_train, y_train)
    score = logreg.score(X_test, y_test)


    print(f'El error para pima es: {1-score}')


    # dataset2 = pd.read_csv('ConjuntosDatosP2/wdbc.csv')
    # X = dataset2[['Atributo1','Atributo2','Atributo3','Atributo4','Atributo5','Atributo6','Atributo7','Atributo8','Atributo9','Atributo10','Atributo11','Atributo12','Atributo13','Atributo14','Atributo15'
    #           ,'Atributo16','Atributo17','Atributo18','Atributo19','Atributo20','Atributo21','Atributo22','Atributo23','Atributo24','Atributo25','Atributo26','Atributo27','Atributo28','Atributo29','Atributo30']].values
    # y = dataset2[['Class']].values.ravel()
    # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000).fit(X, y)
    # clf.predict(X[:, :])
    # print(f'El error para wdbc es: {1-clf.score(X, y)}')

    
    # dataset3 = pd.read_csv('ConjuntosDatosP2/pima-indians-diabetes.csv')
    # X = dataset3[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
    # y = dataset3[['Class']].values.ravel()
    # clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))   
    # clf.fit(X, y)
    # #clf.predict([[-0.8, -1]]
    # print(clf.score(X,y))

    # print(f'El error para pima es: {1-clf.score(X, y)}')

    


if __name__ == "__main__":
    main()