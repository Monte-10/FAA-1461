
from ast import While
from Datos import *
import random
import math
import pandas as pd
from ClasificadorKNN import *
        
def distance(list1,list2):
    """Distance between two vectors."""
    #squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    #return sum(squares) ** .5
    distancia = 0
    total = len(list1) - 1
    for i in range(total):
        x1 = list1[i]
        x2 = list2[i]
        if isinstance(x1, str):
            if isfloat(x1) is True:
                x1 = float(x1)
            else:
                x1 = int(x1)

        if isinstance(x2, str):
            if isfloat(x2) is True:
                x2 = float(x2)
            else:
                x2 = int(x2)
        distancia +=  np.linalg.norm(x1-x2)
    return distancia
    
    

class KMeans:

    def __init__(self, n_clusters=4):
        self.K = n_clusters
        
    def fit(self, datos):
        keys = datos.keys().tolist()
        
        datos[keys] = datos[keys].astype(float)
        #datosNumpy = datos[['SL','SW','PL','PW']].to_numpy() sin la clase
        datosNumpy = datos.to_numpy()
        self.centroids = datosNumpy[np.random.choice(len(datosNumpy), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.clase_anterior,  self.clusterAlQuePertenece = None, np.zeros(len(datosNumpy))
        while not np.all(self.clusterAlQuePertenece == self.clase_anterior) : #cuando no haya ningun cambio en las clusterAlQuePertenece, falta meter el resto de comprobaciones
            self.clase_anterior = self.clusterAlQuePertenece
            self.clusterAlQuePertenece = self.getCentroideMasCercano(datosNumpy)
            self.recalculaCentroides(datosNumpy)
            
        #TODO: el problema que tengo es que la hacerlo numpy, me esta haciendo operaciones sobre la clase, y me esta saliendo con decimales. Entonces no predice bien
        
    def getCentroideMasCercano(self, datosNumpy):
        clases = np.apply_along_axis(self.calculaObjetivo, 1, datosNumpy) #aplica la funcion sobre cada eje del dataset. axis 1 significa operar por cada fila
        #print(f'---- {clases} ----{len(clases)}\n')
        return clases

    def calculaObjetivo(self, x):
        argmin = np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1))) #selecciona el centroide al que menos distancia tiene cada fila
        return argmin #devuelve el centroide que esta a menos distancia respecto a los demas por cada fila

    def recalculaCentroides(self, datosNumpy): #los centroides se calculan para todo xi, calculamos un valor medio nuevo (que no esta en el excel. Y luego se ve cual se acerca más del excel) la clase tambien entra dentro de este calculo.
        for k in range(self.K):
        
            #otra idea seria meter en un numpy datosNumpy[self.clusterAlQuePertenece == k,0:-1], ordenarlo 
            #array = np.array(datosNumpy[self.clusterAlQuePertenece == k])

            
            #array2 = np.sort(array,axis=0)
        

            #Si se puede utilizar un patron medio que no esté en el dataset haremos lo de a continuacion, ya que ofrece mejores resultados.
            #Tras respuesta de mario, nos quedamos con el centro de masas aunque no pertenezca al cluster
            array = np.array(np.mean(datosNumpy[self.clusterAlQuePertenece == k,0:-1],axis=0))
            array2 = np.append(array,round(np.mean(datosNumpy[self.clusterAlQuePertenece == k][:,-1])))
            
            
            
            #self.centroids[k] = array2[math.floor(len(array2)/2)]
            self.centroids[k] = array2
    
    def error(self,datos):
        keys = datos.keys().tolist()
        error = 0
        classesToPredict = []
        for elem in self.centroids:
            classesToPredict.append(round(elem[-1]))
        clases = datos[keys[-1]]
        
        for i in range(len(clases)):
            #print(f'clases[i]: {clases[i]} ===== classesToPredict[self.clusterAlQuePertenece[i]]: {classesToPredict[self.clusterAlQuePertenece[i]]}')
            if clases[i] == classesToPredict[self.clusterAlQuePertenece[i]]:
                pass
            else:
                error += 1
        return error/len(clases)

          
if __name__ == '__main__':

    #print(dataset.datos['Class'])
    #clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
    #clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos
   
    dataset = Datos('ConjuntosDatosP2/iris.csv')
    # otro = ClasificadorKNN().normalize(dataset.datos)
    # dataset.datos = otro
    error = []
        
    for i in range(100):
        dataset = Datos('ConjuntosDatosP2/iris.csv')
        km = KMeans(5)  
        #km.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
        #km.normalizarDatos(dataset.datos,dataset.nominalAtributos)
        # print(type(dataset.datos))
        km.fit(dataset.datos)
        # print(km.clases)
        error.append(km.error(dataset.datos) * 100)
        
    print(np.mean(error))
    #print(km.clusterAlQuePertenece)
    #print(km.centroids)
    
    #print(f'{km.error(dataset.datos) * 100} %')


