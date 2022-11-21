import Clasificador as nb
import ClasificadorKNN as knn
import ClusteringKMeans as kmeans
from Datos import Datos
import EstrategiaParticionado as ep
import RegresionLogistica as rl
import numpy as np

# Creamos una matriz de confusion usando la estrategia de validacion simple y el clasificador Naive Bayes


def matrizConfusionKNN(dataset):
    kNN = knn.ClasificadorKNN()
    matrizConfusion = np.zeros((2,2))
    validacionSimple = ep.ValidacionSimple(25,5)
    validacionSimple.creaParticiones(dataset.datos)
    for i in range(5):
        datosTrain = dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain)
        datosTest = dataset.extraeDatos(validacionSimple.particiones[i].indicesTest)
        # kNN.entrenamiento(datosTrain,dataset.nominalAtributos)
        predicciones = kNN.clasifica(datosTest,datosTrain,dataset.nominalAtributos,5)
        clases = datosTest.iloc[:,-1].to_numpy().astype('int64')
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        counter = 0
        for elem in clases:
            # print(f'{type(elem)}  ============== {type(prediccion[counter])}')
            # print(f'real: {elem}  ============== pred: {prediccion[counter]}')
            # print(type(elem),type(predicciones[counter]))
            if elem == int(predicciones[counter]):
                if elem == 1:
                    # print("tp + 1\n")
                    tp += 1
                else:
                    # print("tn + 1\n")
                    tn += 1
            else:
                if elem == 1:
                    # print("fn + 1\n")
                    fn += 1 
                else:
                    # print("fp + 1\n")
                    fp += 1
            counter += 1

    print(tp,tn,fn,fp)
    return matrizConfusion

matriz = matrizConfusionKNN(Datos('/mnt/c/Users/alexm/Documents/FAA-1461/Practica_3/ConjuntosDatosP2/pima-indians-diabetes.csv'))