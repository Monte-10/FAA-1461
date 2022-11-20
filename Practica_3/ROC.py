import Clasificador as nb
import ClasificadorKNN as knn
import ClusteringKMeans as kmeans
from Datos import Datos
import EstrategiaParticionado as ep
import RegresionLogistica as rl
import numpy as np

# Crea una matriz de confusion para el clasificador Naive Bayes
def matrizConfusionNB(datos, particionado, clasificador, seed=None):
    # Creamos la matriz de confusion
    matriz = np.zeros((len(datos.diccionario['Class']), len(datos.diccionario['Class'])), dtype=int)
    # Creamos las particiones siguiendo la estrategia
    particiones = particionado.creaParticiones(datos.datos, seed)
    # Para cada particion
    for particion in particiones:
        # Entrenamos el clasificador
        clasificador.entrenamiento(datos.extraeDatos(particion.indicesTrain), datos.nominalAtributos, datos.diccionario)
        # Obtenemos las predicciones
        predicciones = clasificador.clasifica(datos.extraeDatos(particion.indicesTest), datos.nominalAtributos, datos.diccionario)
        # Para cada prediccion
        for i in range(len(predicciones)):
            # Aumentamos en uno el valor de la matriz de confusion
            matriz[datos.diccionario['Class'][predicciones[i]]][datos.diccionario['Class'][datos.extraeDatos(particion.indicesTest)[i][-1]]] += 1
    # Devolvemos la matriz de confusion
    return matriz

# Imprime la matriz de confusion para el clasificador Naive Bayes
def imprimeMatrizConfusionNB(datos, particionado, clasificador, seed=None):
    # Obtenemos la matriz de confusion
    matriz = matrizConfusionNB(datos, particionado, clasificador, seed)
    # Imprimimos la matriz de confusion
    print("Matriz de confusion para el clasificador Naive Bayes")
    print("--------------------------------------------------")
    print("Clases: ", datos.diccionario['Class'])
    print("--------------------------------------------------")
    print(matriz)
    print("--------------------------------------------------")

# Imprime TPR, FNR, TNR, FPS para el clasificador Naive Bayes
def imprimeMetricasNB(matriz):
    # Obtenemos los valores de la matriz de confusion
    TP = matriz[0][0]
    FN = matriz[0][1]
    FP = matriz[1][0]
    TN = matriz[1][1]
    # Calculamos las metricas
    TPR = TP / (TP + FN)
    FNR = FN / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    # Imprimimos las metricas
    print("Metricas para el clasificador Naive Bayes")
    print("--------------------------------------------------")
    print("TPR: ", TPR)
    print("FNR: ", FNR)
    print("TNR: ", TNR)
    print("FPR: ", FPR)
    print("--------------------------------------------------")
    
data = Datos('ConjuntoDatosP3/wdbc.csv')
particion = ep.ValidacionSimple(0.7,5)
clasificanb = nb.ClasificadorNaiveBayes()

# Crear una matriz de confusion para el clasificador KNN
def matrizConfusionKNN(datos, particionado, clasificador, seed=None):
    # Creamos la matriz de confusion
    matriz = np.zeros((len(datos.diccionario['Class']), len(datos.diccionario['Class'])), dtype=int)
    # Creamos las particiones siguiendo la estrategia
    particiones = particionado.creaParticiones(datos.datos, seed)
    # Para cada particion
    for particion in particiones:
        # Entrenamos el clasificador
        clasificador.entrenamiento(datos.extraeDatos(particion.indicesTrain), datos.nominalAtributos, datos.diccionario)
        # Obtenemos las predicciones
        predicciones = clasificador.clasifica(datos.extraeDatos(particion.indicesTest), datos.nominalAtributos, datos.diccionario)
        # Para cada prediccion
        for i in range(len(predicciones)):
            # Aumentamos en uno el valor de la matriz de confusion
            matriz[datos.diccionario['Class'][predicciones[i]]][datos.diccionario['Class'][datos.extraeDatos(particion.indicesTest)[i][-1]]] += 1
    # Devolvemos la matriz de confusion
    return matriz

# Imprime la matriz de confusion para el clasificador KNN
def imprimeMatrizConfusionKNN(datos, particionado, clasificador, seed=None):
    # Obtenemos la matriz de confusion
    matriz = matrizConfusionKNN(datos, particionado, clasificador, seed)
    # Imprimimos la matriz de confusion
    print("Matriz de confusion para el clasificador KNN")
    print("--------------------------------------------------")
    print("Clases: ", datos.diccionario['Class'])
    print("--------------------------------------------------")
    print(matriz)
    print("--------------------------------------------------")
    
# Imprime TPR, FNR, TNR, FPS para el clasificador KNN
def imprimeMetricasKNN(matriz):
    # Obtenemos los valores de la matriz de confusion
    TP = matriz[0][0]
    FN = matriz[0][1]
    FP = matriz[1][0]
    TN = matriz[1][1]
    # Calculamos las metricas
    TPR = TP / (TP + FN)
    FNR = FN / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    # Imprimimos las metricas
    print("Metricas para el clasificador KNN")
    print("--------------------------------------------------")
    print("TPR: ", TPR)
    print("FNR: ", FNR)
    print("TNR: ", TNR)
    print("FPR: ", FPR)
    print("--------------------------------------------------")
    
data = Datos('ConjuntoDatosP3/wdbc.csv')
particion = ep.ValidacionSimple(0.7,5)
clasificaknn = knn.ClasificadorVecinosProximos(5)

# Crear una matriz de confusion para el clasificador KMeans
def matrizConfusionKMeans(datos, particionado, clasificador, seed=None):
    # Creamos la matriz de confusion
    matriz = np.zeros((len(datos.diccionario['Class']), len(datos.diccionario['Class'])), dtype=int)
    # Creamos las particiones siguiendo la estrategia
    particiones = particionado.creaParticiones(datos.datos, seed)
    # Para cada particion
    for particion in particiones:
        # Entrenamos el clasificador
        clasificador.entrenamiento(datos.extraeDatos(particion.indicesTrain), datos.nominalAtributos, datos.diccionario)
        # Obtenemos las predicciones
        predicciones = clasificador.clasifica(datos.extraeDatos(particion.indicesTest), datos.nominalAtributos, datos.diccionario)
        # Para cada prediccion
        for i in range(len(predicciones)):
            # Aumentamos en uno el valor de la matriz de confusion
            matriz[datos.diccionario['Class'][predicciones[i]]][datos.diccionario['Class'][datos.extraeDatos(particion.indicesTest)[i][-1]]] += 1
    # Devolvemos la matriz de confusion
    return matriz

# Imprime la matriz de confusion para el clasificador KMeans
def imprimeMatrizConfusionKMeans(datos, particionado, clasificador, seed=None):
    # Obtenemos la matriz de confusion
    matriz = matrizConfusionKMeans(datos, particionado, clasificador, seed)
    # Imprimimos la matriz de confusion
    print("Matriz de confusion para el clasificador KMeans")
    print("--------------------------------------------------")
    print("Clases: ", datos.diccionario['Class'])
    print("--------------------------------------------------")
    print(matriz)
    print("--------------------------------------------------")
    
# Imprime TPR, FNR, TNR, FPS para el clasificador KMeans
def imprimeMetricasKMeans(matriz):
    # Obtenemos los valores de la matriz de confusion
    TP = matriz[0][0]
    FN = matriz[0][1]
    FP = matriz[1][0]
    TN = matriz[1][1]
    # Calculamos las metricas
    TPR = TP / (TP + FN)
    FNR = FN / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    # Imprimimos las metricas
    print("Metricas para el clasificador KMeans")
    print("--------------------------------------------------")
    print("TPR: ", TPR)
    print("FNR: ", FNR)
    print("TNR: ", TNR)
    print("FPR: ", FPR)
    print("--------------------------------------------------")
    
data = Datos('ConjuntoDatosP3/wdbc.csv')
particion = ep.ValidacionSimple(0.7,5)
clasificakmeans = kmeans.KMeans(2)