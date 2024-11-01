from abc import ABCMeta,abstractmethod
from xmlrpc.client import boolean

from zmq import THREAD_AFFINITY_CPU_REMOVE
import numpy as np
import math
import operator
import EstrategiaParticionado
from functools import reduce
from scipy.stats import norm
import statistics

class Clasificador():
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  def __init__(self) -> None:
    pass
    
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
  # datosTest: matriz numpy con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # @args datos: matriz con los datos
  #       pred: predicción 
  def error(self,datos,pred):
    err = 0
    for i in range(datos.shape[0] - 1 ): #numero de elementos de cada atributo
      if datos[datos.keys()[-1]].values[i] != pred[i]:
        err += 1
    
      # if datos['Class'][i] != pred[i]:
      #   err += 1
    #print("Total error:" + str(err) + "total datos: " + str(datos.shape[0]))
    return (err/datos.shape[0])
    
# Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.    
  
  def validacion(self,particionado,dataset,clasificador,seed=None, laPlace = False):
       
    particionado.creaParticiones(dataset.datos, seed)
    errores = []

    for particion in particionado.particiones:
      #extraemos los datos de train y de test de nuestro dataset
      datosTrain = dataset.extraeDatos(particion.indicesTrain)
      datosTest = dataset.extraeDatos(particion.indicesTest)
      #entrenamos nuestro clasificador
      clasificador.entrenamiento(datosTrain, dataset.nominalAtributos,laPlace, dataset.diccionario)
      #realizamos las predicciones y calculamos el error de cada particion.
      predicciones = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionario)
      self.score(datosTest, predicciones)
      errores.append(self.error(datosTest, predicciones))
    #print(statistics.mean(errores))
    return errores                                                    
    
    
 
class ClasificadorNaiveBayes(Clasificador): 
    dicc_atrib = {}
    dicc_clas = {}
    
    
    def __init__(self) :
      super().__init__()
      self.matrizConfusion = np.empty((2,2)) #¿asumo que los valores de clase siempre van a ser 0 y 1? De momento si. Inicializo con tam de matrix 2x2

    def entrenamiento(self,datosTrain ,nominalAtributos, laplace : boolean, datos):
      """Entrenamiento"""
      """obtiene probabilidad a priori y a posteriori en funcion del conjunto de datos de train(datos)"""
      
      self.prioris = {}
      #probCondicionadas = {} #diccionario que contiene {columna:{valor1:{clase1:Nveces,clase2:Nveces},valor2:{clase1:Nveces}}}
      ##print("#Calculamos los prioris#")
      self.prioris,conteoClase = self.getPrioris(datosTrain[datosTrain.keys()[-1]], nominalAtributos[-1],laplace)#asumimos que la clase es el ultimo atributo siempre

      ##print("Prioris: " + str(self.prioris))
      
      clases = np.unique(datosTrain[datosTrain.keys()[-1]])
      self.tablaSolucion = [] #contiene todas una lista por cada atributo con el nnumero de aparicioines de la clase por cada valor del atributo
      counter = 0
      for key,values in datos.items():
        #para no coger la clase
        if counter == len(datos) -1:
          break
        if nominalAtributos[counter]:
          #print("Creo tabla")
          tabla = np.zeros((len(values),len(clases)))
          for index,row in datosTrain.iterrows():
            tabla[int(row[counter])-1,int(row[-1])-1] += 1

          if laplace is True:
            tabla += 1
          
        else:
          #print("Creo  tabla por no nominal")
          tabla = np.zeros((2,len(clases)))  #[mediaClase1][varianzaClase1]
                                             #[mediaClase2][varianzaClase2]
          
          auxCont = 0
          for clase in clases:
            lista = []
            for index,row in datosTrain.iterrows():
              if(row[-1] == clase):
                lista.append(float(row[counter]))
            
            media = np.mean(lista)
            varianza = np.var(lista)
            tabla[0][auxCont] = media
            tabla[1][auxCont] = varianza
            auxCont += 1
          
        counter += 1
        self.tablaSolucion.append(tabla)
    def clasifica(self,datosTest,nominalAtributos,dicc):
      
      pred = []
      
      ##print(datosTest)
      for index,fila in datosTest.iterrows():
        
        ##print(fila)
        prodHi = {}

        contClases = 0
        for priori in self.prioris: 
          pHi = self.prioris[priori] 
          ##print("pHi ->" + str(pHi))
          j = 0
          prod = pHi
          ##print(len(datosTest.keys()))
          while j < len(datosTest.keys()) - 1:
            ##print("J -> " + str(j))
            if nominalAtributos[j]:
              ##print(self.tablaSolucion[j])  
              ##print(self.tablaSolucion[j][:, contClases])
              ejsClase = sum(self.tablaSolucion[j][:, contClases])
              ##print(ejsClase)
              ##print("Fila[j] -> " + str(fila[j]) )
              ##print("tablasolcion[j][Fila[j]] -> " + str(fila[j]) )
              prod *= self.tablaSolucion[j][int(fila[j])-1][contClases] / ejsClase 
            else:
              op1 = (1/(math.sqrt(2*math.pi*self.tablaSolucion[j][1][contClases])))
              op2 = math.exp((-((float(fila[j]))-self.tablaSolucion[j][0][contClases]))/(2*self.tablaSolucion[j][1][contClases]))
              
              prod *= op1*op2

            j += 1
          prodHi[priori] = prod
          contClases += 1
        
        pred.append(max(prodHi, key=prodHi.get)) #seleccionamos el maximo de cada producto de hi
      
      return pred
    
    def getPrioris(self, datosTrain, nominalAtributo, laplace):
      diccionario = {}
      
      for elem in datosTrain:
        if diccionario.__contains__(elem):
          diccionario[elem] += 1
        else:
          diccionario[elem] = 1

      total = len(datosTrain)
      diccionarioSolucion = {}
      for elem in diccionario.keys():
        diccionarioSolucion[elem] = diccionario[elem] / total 
    
      return diccionarioSolucion,diccionario
    
      
    '''
    Sera invocada de forma iterativa, por cada columna. 
    De esta forma, calculara para cada valor que se encuentre la probabilidad condicionada por cada valor en el diccionario de prioris. 
    Es decir p(A1=1 | C=1), p(A1=2 | C=1), p(A1=1 | C=2) , p(A1=2 | C=2) y todas posibles combinaciones

    clase -> lista con todas las filas de la columna clase.
    conteoClase -> numeroApariciones para cada valor de la clase
    '''
    def getProbabilidadesCondicionadas(self, prioris, datosTrain, nominalAtributo,clase,conteoClase,laplace): 

      valoresClase = []
      miLista = []
      for valor in clase:
        miLista.append(valor)
        if not valoresClase.__contains__(valor):
          valoresClase.append(valor)
        
      #diccionarioFinal -> {atr = 0:{clase = 1:}, atr = 1{clase = 1:,clase = 2:,clase = 3:}}
      if(nominalAtributo is True):
        diccionarioSolucion = {}
        i = 0
        for elem in datosTrain: #diccionario con el conteo de todos los valores de la columna          
          if diccionarioSolucion.__contains__(elem) and diccionarioSolucion[elem].__contains__(miLista[i]):            
            diccionarioSolucion[elem][miLista[i]] += 1
          else:
            if diccionarioSolucion.__contains__(elem):
              diccionarioSolucion[elem][miLista[i]]= 1
            else:   
              diccionarioSolucion[elem] = {}
              diccionarioSolucion[elem][miLista[i]] = 1 
        
          i += 1
        

        '''CORRECION DE LAPACE'''
        if(laplace is True):  
          ##print("Correccion de laplace")
          for elem in diccionarioSolucion.keys():
            for clase in diccionarioSolucion[elem].keys():
              diccionarioSolucion[elem][clase] += 1
          for elem in conteoClase.keys():
            conteoClase[elem] += 1
        '''----------------------'''
        diccionarioFinal = {}

        for elem in diccionarioSolucion.keys():
          diccionarioFinal[elem] = {}
          for clase in diccionarioSolucion[elem].keys():
            diccionarioFinal[elem][clase] = diccionarioSolucion[elem][clase] / conteoClase[clase] 
          
        i = 0
        return diccionarioFinal
      else:
        miDict = {}
        for e in clase:
          ##print(e.index[0])
          if miDict.__contains__(e):            
            miDict[e].append(int(datosTrain[clase.index[0]]))
          else:
            miDict[e] = []
            miDict[e].append(int(datosTrain[clase.index[0]]))

        '''{clase=N : [1,4,2,...,N]}'''
        diccionarioSolucion = {}
        for elem in miDict.keys():
          lista = miDict[elem]
          '''media'''
          mean = sum(lista) / len(lista)
          var = sum((l-mean)**2 for l in lista) / len(lista)
          '''desviacion estandar'''
          st_dev = math.sqrt(var)          
          diccionarioSolucion[elem] = (mean,st_dev)
        return diccionarioSolucion
        '''{clase = 1: (media,desviacion tipica),clase = 2: (media,desviacion tipica)}'''
        '''calcular media y desviacion tipica para cada valor de la clase '''
  
    def score(self,datosTest,prediccion):
        clases = datosTest.iloc[:,-1].to_numpy().astype('int64')

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        counter = 0
        for elem in clases:
            if elem == int(prediccion[counter]):
                if elem == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if elem == 1:
                    fn += 1 
                else:
                    fp += 1
            counter += 1

        self.matrizConfusion[0][0] = int(tp)
        self.matrizConfusion[0][1] = int(fp)
        self.matrizConfusion[1][0] = int(fn)
        self.matrizConfusion[1][1] = int(tn)

        self.TPR = tp / (tp+fn)
        self.FNR = fn / (tp+fn)
        self.FPR = fp / (fp+tn)
        self.TNR = tn / (fp+tn)
        

def dist_normal(m,v,n):
      if (v == 0):
        v += math.pow(10, -6)

      exp = -(math.pow((n-m), 2)/(2*v))
      base = 1/math.sqrt(2*math.pi*v)
      densidad = base*math.pow(math.e,exp)
      return densidad

 