
from ast import While
from Datos import *
import random
import math
import pandas as pd

class Kmeans2: 
    
    #k -> nº clusters
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = []
        self.idx = []
        self.stoppingCriteria = False
        self.maximoCambioCentroids = 10
        self.centroids = []
        
    '''almacena en self.idx[0->k] el indice pandas que hace referencia al centroide del cluster. Esto es util para posteriormente obtener del dataset la fila completa y el valor de la clase
                    self.cluster[0->k]'''    
    def getRandomClusters(self):
        index = self.data.index.tolist()
        random.shuffle(index)
        for i in range(self.k):
            print(f"el centroide del cluster {i} va a ser el indice {index[i:i+1]}")
            self.idx.append(index[i:i+1]) #contiene el indice pandas que hace referencia al centroide del cluster
            self.centroids.append(np.take(self.data, self.idx[-1],axis=0).squeeze()) #self.clusters[i] -> contiene la informacion del centroide de cada cluster
            self.clusters.append([])
            self.clusters[-1].append(self.centroids[-1]) # metemos en el ultimo cluster, el ultimo centroide    
    
    def casteaSeries(self,cluster):
        for elem in cluster:
            if isfloat(elem) is True:

                elem = float(elem)

            else:

                elem = int(elem)
    def casteaNumpy(self,array):
        miarray = np.zeros(0)
        for elem in array:
            if isinstance(elem, str):                
                if isfloat(elem) is True:
                    miarray = np.append(miarray,float(elem))  

                else:
                    miarray = np.append(miarray,int(elem)) 
                    
            else:
                miarray = np.append(miarray,elem) 
                
        return miarray


    
            
    def recalculoCentroide(self): #al salir de esta funcion, self.clusters debe quedar con un nuevo elemento en cada uno que es el centroide del cluster, y self.idx con el indice de este nuevo cluster
        #self.idx los centoides
        #self.clusters los cluster
        nparray= []
        for i in range(self.k):
            DF = pd.DataFrame()
            for elem in self.clusters[i]:
                elemnumpy = elem.to_numpy()
                
                nparray.append(self.casteaNumpy(elemnumpy))
            print(nparray)
            cm = np.average(nparray[:,:], axis=0) # CON ESTO OBTENEMOS EL VECTOR MEDIA.
            print(cm)
            # for elem in self.clusters[i]: 
                







            break

        
        
        
        
        
        # index = self.data.index.tolist()
        # keys = self.data.keys().tolist()
        # self.maximoCambioCentroids -= 1
        # if self.maximoCambioCentroids < 1:
        #     self.stoppingCriteria = True
        # miCluster = np.take(self.data, self.idx[0],axis=0).squeeze()
        # print(keys)
        # miCluster[keys[:-1]] = miCluster[keys[:-1]].astype(float)
        
        # miClusterNP = miCluster.to_numpy()
        # #self.castea(miClusterNP)
        # print(miClusterNP)
        
        # # print(miClusterNP)
        # cm = np.average(miClusterNP[:,:], axis=0) # CON ESTO OBTENEMOS EL VECTOR MEDIA.
        # print(cm)
        # shortestDistance = 99999
        # shortestIndex = []
        # for elem in miClusterNP:
        #     distancia = distance(cm[:-1],elem[:-1])
        #     if distancia < shortestDistance:
        #         shortestDistance = distancia
            
        #     # distance(cm[:-1],elem[:-1]) #la clase no se utiliza para calcular la distancia
        #     shortestDistance.append(distance(cm[:-1],elem[:-1])) #la clase no se utiliza para calcular la distancia)
            
        # print(min(shortestDistance))
        
        # #print(f'He entrado en recalculo y este es el valor de la fila {self.data.loc[self.idx[0][0].squeeze()]}')

        
        # for i in range(self.k):
        #     #obtener el elemento medio del cluster y elegirlo como el nuevo centoride de este.
        #     print(self.idx[i])

        #     nuevoIndice = math.floor(len(self.idx[i]))
        #     print(f'El indice medio : {nuevoIndice}')
        #     self.idx[i] = []
        #     self.idx[i].append(nuevoIndice) 
        #     self.clusters[i] = np.take(self.data, index[nuevoIndice:nuevoIndice+1],axis=0).squeeze()
        #     print(f'self.idx[i] {self.idx[i]} deberia ser igual que indicemedio ¿lo es?')
        #     print(f'self.clusters[i]: {self.clusters[i]} debe corresponder a la fila de index')

        
    #por cada fila calculamos la distancia con cada cluster    
    def getCentroideMasCercano(self):
       
        #print(self.clusters)
        
        for index,datosTabla in self.data.iterrows():
            self.casteaSeries(datosTabla)
            contador = 0
            distanciasVsCluster = []
            flag = 0
            for elem in self.idx:
                if index == elem[0]:
                    flag = 1
            if flag == 0:
                for centroide in self.centroids:                    
                        self.casteaSeries(centroide)
                    #el indice del centroide no es el mismo que el que vamos a calcular la distanca. porque sino, se estaria comparando consigo mismo.
                        #print(f'tipo de datos tabla {type(datosTabla)}')        
                        #print(f'tipo de infoCluster {type(infoCluster)}')        
                        distanciasVsCluster.append((centroide.name,distance(datosTabla,centroide))) # de esta forma tenemos en distanciasVsCluster[i] una tupla (indice, distancia)
                        #print(f'La distancia del {index} con el cluster {self.idx[contador]} es: {distanciasVsCluster[contador]}')
                    
                contador += 1  
                minimo = min(distanciasVsCluster, key = lambda t: t[1]) # minimo contiene la distancia y el nombre del centroide
                  

                #print(f'La minima distancia entre clusters para el indice {index}, corresponde al cluster {minimo[0]}')
                flag = 0
                for lista in self.clusters:
                    for elem in lista:
                        if minimo[0] == elem.name:
                            #print(f'{elem.name} vs {minimo[0]}')
                            lista.append(datosTabla) 
                            flag = 1
                            break
                    if flag == 1:
                        break
                else:
                    pass
        
        
    def fit(self):
        #choose k data points as the initial centroids (cluster centers)
        self.getRandomClusters()
        while self.stoppingCriteria is False:
            self.getCentroideMasCercano()
            self.recalculoCentroide()
            break #TODO:eliminar break para hacer algoritmo completo


        
        
        
            
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
        #X = datos[['SL','SW','PL','PW']].to_numpy() sin la clase
        X = datos.to_numpy()
        print(X)
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        while not np.all(self.labels == self.prev_label) : #cuando no haya ningun cambio en las clases
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
        return self
        
    def predict(self, X):
        return np.apply_along_axis(self.compute_label, 1, X) #aplica una funcion sobre un eje. axis es 1 porque datos es [[]], entonces 1 apunta a dentro del primer array

    def compute_label(self, x):
        argmin = np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))
        return argmin

    def update_centroid(self, X): #los centroides se calculan para todo xi, calculamos un valor medio nuevo (que no esta en el excel. Y luego se ve cual se acerca más del excel) la clase tambien entra dentro de este calculo.
        
        for k in range(self.K):
            self.centroids[k] = np.array(np.mean(X[self.labels == k], axis=0))
        

    
if __name__ == '__main__':

    dataset = Datos('ConjuntosDatosP2/iris.csv')
    print(dataset.datos['Class'])
    km = KMeans(3)  
    # print(type(dataset.datos))
    km.fit(dataset.datos)
    # print(km.labels)
    print(km.labels)
    print(km.centroids)
    # km.getCentroideMasCercano()

