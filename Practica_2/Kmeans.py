
from ast import While
from Datos import *
import random
import math
import pandas as pd
        
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
        clases = np.apply_along_axis(self.compute_label, 1, datosNumpy) #aplica la funcion sobre cada eje del dataset. axis 1 significa operar por cada fila
        #print(f'---- {clases} ----{len(clases)}\n')
        return clases

    def compute_label(self, x):
        argmin = np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1))) #selecciona el centroide al que menos distancia tiene cada fila
        return argmin #devuelve el centroide que esta a menos distancia respecto a los demas por cada fila

    def recalculaCentroides(self, datosNumpy): #los centroides se calculan para todo xi, calculamos un valor medio nuevo (que no esta en el excel. Y luego se ve cual se acerca más del excel) la clase tambien entra dentro de este calculo.
        for k in range(self.K):
            # print(f'K --------->{k}')
            # print(f'----- clusterAlQuePertenece: {self.clusterAlQuePertenece}')
            # print(f'----- datosNumpy[:-1]: {datosNumpy[self.clusterAlQuePertenece == k][:,:-1]}')
            # print(f'----- datosNumpy[:-1]: {datosNumpy[self.clusterAlQuePertenece == k][:-1].mean(axis=0)}')
            # print(f'{datosNumpy[self.clusterAlQuePertenece == k]}') #devuelve las posiciones de la tabla que pertenecen al cluster en cuestion
             
            array = np.array(np.mean(datosNumpy[self.clusterAlQuePertenece == k],axis=0))
           
            #array2 = np.append(array,datosNumpy[self.clusterAlQuePertenece == k][:,-1])
            
            #array2 = np.array(array) #TODO: el nuevo centroide es la media(CENTRO DE MASAS) de cada cluster. Pero tiene que ser necesariamente un valor que estuviera previamente en el cluster. O puede ser otro nuevo
            
            
            
            self.centroids[k] = array
    
    def error(self,datos):
        keys = datos.keys().tolist()
        error = 0
        classesToPredict = []
        for elem in self.centroids:
            classesToPredict.append(round(elem[-1]))
        clases = datos[keys[-1]]
        
        for i in range(len(clases)):
            if clases[i] == classesToPredict[self.clusterAlQuePertenece[i]]:
                pass
            else:
                error += 1
        return error/len(clases)

    def calcularMediaDesviacion(self,datos,nominalAtributos):
      lista = []
      for elem in datos.values:
          for i in elem:
              if isfloat(i):
                  lista.append(float(i))
              else:
                  lista.append(int(i))
      self.mediaNormalizacion = np.mean(lista)
      self.desviacionTipicaNormalizacion = np.std(lista)

    #formula = (Xi - mu / desv)
    def normalizarDatos(self,datos,nominalAtributos):
        i = 0
    

        for elem in datos.keys():
            if elem is not datos.keys()[-1]:
                lista = []
                if nominalAtributos[i] is False:
                    for value in datos[elem]:
                        if value not in lista:
                            normalizado = (float(value) - self.mediaNormalizacion) / self.desviacionTipicaNormalizacion
                            datos[elem] = datos[elem].replace([value],normalizado)
                            lista.append(value)
                i +=1


        
if __name__ == '__main__':

    #print(dataset.datos['Class'])
    #clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
    #clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos
    error = []
   
    dataset = Datos('ConjuntosDatosP2/iris.csv')
    km = KMeans(3)  
    km.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
    km.normalizarDatos(dataset.datos,dataset.nominalAtributos)
    # print(type(dataset.datos))
    km.fit(dataset.datos)
    # print(km.clases)
    # print(km.clusterAlQuePertenece)
    # print(km.centroids)
    print(f'{km.error(dataset.datos) * 100}%')


