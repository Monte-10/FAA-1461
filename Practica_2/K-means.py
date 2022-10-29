
from Datos import *
import random
import math

class Kmeans2: 
    
    #k -> nº clusters
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = []
        self.idx = []
        self.stoppingCriteria = False
        self.maximoCambioCentroids = 10
        
    def getRandomClusters(self):
        index = self.data.index.tolist()
        random.shuffle(index)
        for i in range(self.k):
            self.idx.append([])
            self.idx[i] = index[i:i+1] #contiene el indice de pandas que hace referencia al centroide del cluster
            self.clusters.append([])
            self.clusters[i] = np.take(self.data, index[i:i+1],axis=0).squeeze()
    
    def castea(self,cluster):
        for elem in cluster:
            for elem2 in elem:
                if isfloat(elem2):
                    elem2 = float(elem2)
                else:
                    elem2 = int(elem2)
        
            
    def recalculoCentroide(self): #al salir de esta funcion, self.clusters debe quedar con un nuevo elemento en cada uno que es el centroide del cluster, y self.idx con el indice de este nuevo cluster
        index = self.data.index.tolist()
        keys = self.data.keys().tolist()
        self.maximoCambioCentroids -= 1
        if self.maximoCambioCentroids < 1:
            self.stoppingCriteria = True
        miCluster = np.take(self.data, self.idx[0],axis=0).squeeze()
        print(keys)
        miCluster[keys[:-1]] = miCluster[keys[:-1]].astype(float)
        
        miClusterNP = miCluster.to_numpy()
        #self.castea(miClusterNP)
        print(miClusterNP)
        
        # print(miClusterNP)
        CM = np.average(miClusterNP[:,:], axis=0) # CON ESTO OBTENEMOS EL VECTOR MEDIA.
        #print(CM)
        
        
        miLista = []
        #print(f'He entrado en recalculo y este es el valor de la fila {self.data.loc[self.idx[0][0].squeeze()]}')

        
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
        
        clase = self.data.keys()[-1]
        while self.stoppingCriteria is False:
            for index,elem in self.data.iterrows():
                distanciaCluster = []
                contador = 0
                for cl in self.clusters: 
                    lista1 = []
                    lista2 = []
                    #print(f' {type(self.idx[contador])} , {type(index)}')
                    # if str(self.idx[contador]) != str(index): #necesito comparar por indices, no por contenido. Porque puede ser que coincidan.
                    if index not in self.idx[contador]: #necesito comparar por indices, no por contenido. Porque puede ser que coincidan.

                        #print(f'if {index} not in {self.idx[contador]}  OK' )
                    
                        for i in elem.items(): #datos
                            if i[0] != clase: #la distancia no la calculamos con la clase. Sino con el resto de los atributos
                                if isfloat(i[1]):
                                    lista1.append(float(i[1]))
                                else:
                                    lista1.append(int(i[1]))
                        for i in cl.items(): #clusters
                            if i[0] != clase:
                                if isfloat(i[1]):
                                    lista2.append(float(i[1]))
                                else:
                                    lista2.append(int(i[1]))
                        #print(f'{lista1} - {lista2}' )
                        dist = distance(lista1,lista2)
                        #print(dist)
                        distanciaCluster.append(dist) #lista con k valores, que contiene la distancia de una fila a cada cluster. Seleccionaremos la menor, y asignaremos a ese cluster, esta fila.
                    contador += 1
                print(distanciaCluster)
                index_min = min(range(len(distanciaCluster)), key=distanciaCluster.__getitem__)#TODO: no esta haciendo bien esto, mejor meterlo en una lista y hacer sort, como en knn
                self.idx[index_min].append(index) #guardamos el indice de la fila que estamos calculando la distancia en la posicion que corresponde a al cluster con menor distancia. 
            print(f'Despues de calcular todas las filas, los indices de cada cluster quedan de la siguiente manera:\n{self.idx}')
            self.recalculoCentroide()
            
            break
        #TODO: recalculo del centroide para cada cluster.
                
   



        
        
        
            
def distance(list1,list2):
    """Distance between two vectors."""
    #squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    #return sum(squares) ** .5
    distancia = 0
    total = len(list1)
    for i in range(total):
        distancia +=  np.linalg.norm(list1[i]-list2[i])
    return distancia
    

class KMeans:
    
    def __init__(self, n_clusters=4):
        self.K = n_clusters
        
    def fit(self, datos):
        keys = datos.keys().tolist()
        datos[keys] = datos[keys].astype(float)
        X = datos.to_numpy()
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        while not np.all(self.labels == self.prev_label) :
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
        return self
        
    def predict(self, X):
        
        return np.apply_along_axis(self.compute_label, 1, X)

    def compute_label(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))

    def update_centroid(self, X):
        self.centroids = np.array([np.mean(X[self.labels == k], axis=0)  for k in range(self.K)])

if __name__ == '__main__':

    dataset = Datos('ConjuntosDatosP2/iris.csv')
    km = KMeans(3)  
    print(type(dataset.datos))
    km.fit(dataset.datos)
    print(km.labels)
    #km.getRandomClusters() #TODO: ¿cada cluster contiene varios patrones?
    #km.getCentroideMasCercano()


    # import numpy
    # masses = numpy.array([[0,  0,  0,  0],
    #     [0,  1,  0,  0],
    #     [0,  2,  0,  0],
    #     [1,  0,  0,  0],
    #     [1,  1,  0,  1],
    #     [1,  2,  0,  1],
    #     [2,  0,  0,  0],
    #     [2,  1,  0,  0],
    #     [2,  2,  0,  0]])

    # nonZeroMasses = masses[numpy.nonzero(masses[:,3])] # Not really necessary, can just use masses because 0 mass used as weight will work just fine.

    # CM = numpy.average(masses[:,:], axis=0) # CON ESTO OBTENEMOS EL VECTOR MEDIA.
    # #Falta calcular la distancia para todos los patrones del cluster y el centroide media
    # print(CM)