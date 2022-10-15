from abc import ABCMeta,abstractmethod
from xmlrpc.client import boolean
import numpy as np
import math
import operator
import EstrategiaParticionado
from Datos import Datos
from functools import reduce
from scipy.stats import norm

class Clasificador:
  
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
    
    for i in range(datos.datos.shape[0]):
      if datos[i][-1] != pred[i]:
        err += 1
        
    return (err/datos.shape[0])
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  def validacion(self,particionado,dataset,clasificador,seed=None, laPlace = False):
       
    particionado.creaParticiones(dataset, seed)
    errores = []

    for i in particionado.particiones:
      datosTrain = dataset.extraeDatos(i.indicesTrain)
      datosTest = dataset.extraeDatos(i.indicesTest)

      clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionario)
      predicciones = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionario)
      
      errores.append(self.error(datosTest, predicciones))

    return errores                                                    
    
    
 
class ClasificadorNaiveBayes(Clasificador): 
    dicc_atrib = {}
    dicc_clas = {}
    
    
    def __init__(self) :
      #RTM quiza necesite recibir algo en el constructor (por ejemplo el objeto datos)
      super().__init__()

    def entrenamiento(self,datosTrain : Datos ,nominalAtributos, laplace : boolean):
      """Entrenamiento"""
      """obtiene probabilidad a priori y a posteriori en funcion del conjunto de datos de train(datos)"""
      prioris = {}
      probCondicionadas = {} #diccionario que contiene {columna:{valor1:{clase1:Nveces,clase2:Nveces},valor2:{clase1:Nveces}}}
      prioris,conteoClase = self.getPrioris(datosTrain[datosTrain.keys()[-1]], nominalAtributos[-1],laplace)#asumimos que la clase es el ultimo atributo siempre
      
      print("getPriorisOk:")
      print("Prioris: " + str(prioris))
      i = 0
      # while i < len(datosTrain.keys()) -1: #-1 porque la clase no la voy a mandar
      #   probCondicionadas[i] = self.getProbabilidadesCondicionadas(prioris, datosTrain[datosTrain.keys()[i]], nominalAtributos[i],datosTrain[datosTrain.keys()[-1]],conteoClase,laplace) 
      #   i += 1
      probCondicionadas[i] = self.getProbabilidadesCondicionadas(prioris, datosTrain[datosTrain.keys()[1]], nominalAtributos[1],datosTrain[datosTrain.keys()[-1]],conteoClase,laplace) 
      
      print("\n\n\nCondicionadas: " + str(probCondicionadas))



    def clasifica(self,datosTest,atributos,dicc):
      pred = []

      for fila in datosTest:
        # Calculamos PRODj [(P(Xj|Hi)*P(Hi))] para cada clase
        prodHi = {}

        contClases = 0
        for i in self.tablaAPriori: 
          pHi = self.tablaAPriori[i] # P(Hi)

          j = 0
          prod = pHi
          while j < len(fila) - 1:
            if atributos[j]:
              #print(self.tablasAtributos[j])  
              ejsClase = sum(self.tablasAtributos[j][:, contClases])
              prod *= self.tablasAtributos[j][int(fila[j])][contClases] / ejsClase # P(X1|Hi) * P(X2|Hi) * ...

            else:
              op1 = (1/(math.sqrt(2*math.pi*self.tablasAtributos[j][1][contClases])))
              op2 =math.exp((-(fila[j]-self.tablasAtributos[j][0][contClases]))/(2*self.tablasAtributos[j][1][contClases]))
              prod *= op1*op2

            j += 1
          
          prodHi[i] = prod
          contClases += 1
        
        pred.append(max(prodHi, key=prodHi.get)) # Decision = argmax_Hi PRODj [(P(Xj|Hi)*P(Hi))]
      
      return pred
    
    def getPrioris(self, datosTrain, nominalAtributo, laplace):
      diccionario = {}
      
      #if(nominalAtributo is True): #es nominal
      #creo que los prioris es indiferente si el atributo es nominal o no
      for elem in datosTrain:
        if diccionario.__contains__(elem):
          diccionario[elem] += 1
        else:
          diccionario[elem] = 1
    
      '''Correcion de laplace'''
      #TODO: confirmar que se hace tambien en los prioris
      if(laplace is True):
        for e in diccionario.keys():
          diccionario[e] += 1
      '''--------------------'''
    
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
      diccionario = {}
      counter = 0
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
          print("Correccion de laplace")
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
        counter = 0
        miDict = {}
        for e in clase:
          miDict[e] = []
          miDict[e].append(int(datosTrain[counter]))
          counter += 1
        print(miDict)
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
        print(diccionarioSolucion)
        return diccionarioSolucion
        '''{clase = 1: (media,desviacion tipica),clase = 2: (media,desviacion tipica)}'''
        '''calcular media y desviacion tipica para cada valor de la clase '''
        print(type(datosTrain))
        print(datosTrain)
        media = datosTrain.mean()
        #varianza = datosTrain.std(ddof=0)
        varianza = 0
        print(media)
        return (media,varianza)




      pass

def dist_normal(m,v,n):
      if (v == 0):
        v += math.pow(10, -6)

      exp = -(math.pow((n-m), 2)/(2*v))
      base = 1/math.sqrt(2*math.pi*v)
      densidad = base*math.pow(math.e,exp)
      return densidad

 
    
  

##############################################################################


