
from Datos import *
import random

class Kmeans: 
    
    #k -> nº clusters
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = []
        
    def getRandomClusters(self):
        # lista = []
        # index = self.data.index
        # for i in index:
        #     lista.append(i)
        # random.shuffle(lista)
        # nCluster = len(self.data)/self.k
        # cluster = []
        # i = 0
        # print(lista)
        # for value in range(self.k):
        #     hasta = int(nCluster) + i
        #     print(int(hasta))
        #     cluster.append([])
        #     cluster[value] = lista[i:int(hasta)]
        #     i+=int(nCluster)
        # #TODO: hay que tener cuidado porque como divides el dataset en partes iguales si fuera impar?        
        lista =  []
        index = self.data.index
        for i in index:
            lista.append(i)
        random.shuffle(lista)
        for i in range(self.k):
            self.clusters.append([])
            self.clusters[i] = np.take(self.data, lista[i:i+1],axis=0).squeeze()
    #por cada fila calculamos la distancia por cada cluster    
    def getCentroideMasCercano(self):
        print(self.data[:])
        for index,elem in self.data.iterrows():
            for cl in self.clusters:
                print(cl)
                lista1 = []
                lista2 = []
                for i in elem.items():
                    if isfloat(i[1]):
                        lista1.append(float(i[1]))
                    else:
                        lista1.append(int(i[1]))
                for i in cl.items():
                    if isfloat(i[1]):
                        lista2.append(float(i[1]))
                    else:
                        lista2.append(int(i[1]))
                dist = distance(lista1,lista2)
                print(dist)
                break
            break
        #FALTA NO SUMAR LA CLASE
        #CUANDO QUITE LOS BREAKS FALTARÁ CHEKEAR QUE NO ESTÁ HACIENDO LA DISTANCIA CONSIGO MISMO PORQUE SINO SIEMPRE COGERÁ ESE COMO EL MENOR

        
        
        
            
def distance(list1,list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5


if __name__ == '__main__':

    dataset = Datos('ConjuntosDatosP2/iris.csv')
    km = Kmeans(3,dataset.datos)  
    km.getRandomClusters()
    km.getCentroideMasCercano()