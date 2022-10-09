from abc import ABCMeta,abstractmethod
from xmlrpc.client import boolean
import numpy as np
import math
import operator
import EstrategiaParticionado
from Datos import Datos
from functools import reduce

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
  """def validacion(self,particionado,dataset,clasificador,seed=None, laPlace = False):
       
    particionado.creaParticiones(dataset.datos,None)
    error = []
    data_d = dataset.diccionario
    data_a_d = dataset.nominalAtributos

    if (particionado.nombreEstrategia == "ValidacionSimple"):
      clasificador.entrenamiento(datostrain=dataset.extraeDatos(particionado.listaParticiones[0].indicesTrain)
      atributosDiscretos = data_a_d,
      diccionario = data_d,
      laPlace = laPlace)
      prediccion = clasificador.clasifica(datostest=dataset.extraeDatos(particionado.listaParticiones[0].indicesTest),
                               atributosDiscretos=data_a_d,
                               diccionario=data_d)

      error.append(clasificador.error(datos=dataset.extraeDatos(particionado.listaParticiones[0].indicesTest),
                                    pred=prediccion))

    elif (particionado.nombreEstrategia == "ValidacionCruzada"):
        for i in range(particionado.numeroParticiones):
            clasificador.entrenamiento(datostrain=dataset.extraeDatos(particionado.listaParticiones[i].indicesTrain),
                                   atributosDiscretos=data_a_d,
                                   diccionario=data_d,
                                   laPlace = laPlace)
            prediccion = clasificador.clasifica(datostest=dataset.extraeDatos(particionado.listaParticiones[i].indicesTest),
                                   atributosDiscretos=data_a_d,
                                   diccionario=data_d)
            error.append(clasificador.error(datos=dataset.extraeDatos(particionado.listaParticiones[i].indicesTest),
                                        pred=prediccion))
    
    
    return error, prediccion                             
  """                               
    
    
 
class ClasificadorNaiveBayes(Clasificador): 
    dicc_atrib = {}
    dicc_clas = {}
    
    
    def __init__(self) :
      #RTM quiza necesite recibir algo en el constructor (por ejemplo el objeto datos)
      super().__init__()

    def entrenamiento(self,datosTrain : Datos ,nominalAtributos,laplace : boolean):
      """Entrenamiento"""
      """obtiene probabilidad a priori y a posteriori en funcion del conjunto de datos de train(datos)"""
      prioris = {}
      probCondicionadas = {}
      prioris,conteoClase = self.getPrioris(datosTrain[datosTrain.keys()[-1]], nominalAtributos[-1])#asumimos que la clase es el ultimo atributo siempre
      
      print("getPriorisOk:")
      print("Prioris: " + str(prioris))
      i = 0
      #while i < len(datosTrain.keys()) -1: #-1 porque la clase no la voy a mandar
      # probCondicionadas["at1"] = 
      self.getProbabilidadesCondicionadas(prioris, datosTrain[datosTrain.keys()[i]], nominalAtributos[i],datosTrain[datosTrain.keys()[-1]],conteoClase) #TODO: falta aniadir al diccionario por cada atributo del excel
      i += 1
      
      
      #print("\n\n\nCondicionadas: " + str(probCondicionadas))



    def clasifica(self,datosTest,atributos,dicc):
      post = {}
      pri = {}
      lista = []
      bayes = []
      
      n = sum(list(self.dicc_clas.values()))
      
      for i, j in self.dicc_clas.items():
        pri.update({i:(j/n)})
        
      for cont in range(len(datosTest)):
        
        post.update({cont:{}})
        for i, j in self.dicc_clas.items():
          
          for cont2 in range(datosTest.shape[1] -1):
            
            if 'm' in self.dicc_atrib[cont2].keys():
              m = self.dicc_atrib[cont2]['m'][i]
              var = self.dicc_atrib[cont2]['v'][i]
              gauss = dist_normal(m,var,datosTest[cont][cont2])
              lista.append(gauss)
            
            else:
              c = datosTest[cont][cont2]
              lista.append(self.dicc_atrib[cont2][c][i] / float(i))
        
        bayes.append(reduce(lambda x, y: x*y, lista)*pri[i])
        post[cont][i] = bayes
      pred = np.zeros(datosTest.shape[0])
      for i in range(datosTest.shape[0]):
        pred[i] = max(post[i].items(), key = operator.itemgetter(1))[0]

      return pred
    
    def getPrioris(self, datosTrain, nominalAtributo):
      diccionario = {}
      print(nominalAtributo)
      if(nominalAtributo is True): #es nominal
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
      
      else:
        #TODO: falta hacer el caso en que no es nominal y el caso en el que le llega laplace
        pass
    
    '''
    Sera invocada de forma iterativa, por cada columna. 
    De esta forma, calculara para cada valor que se encuentre la probabilidad condicionada por cada valor en el diccionario de prioris. 
    Es decir p(A1=1 | C=1), p(A1=2 | C=1), p(A1=1 | C=2) , p(A1=2 | C=2) y todas posibles combinaciones

    clase -> lista con todas las filas de la columna clase.
    conteoClase -> numeroApariciones para cada valor de la clase
    '''
    def getProbabilidadesCondicionadas(self, prioris, datosTrain, nominalAtributo,clase,conteoClase): 
      diccionario = {}
      counter = 0
      valoresClase = []
      miLista = []
      for valor in clase:
        miLista.append(valor)
        if not valoresClase.__contains__(valor):
          valoresClase.append(valor)
        
      
      #diccionarioFinal -> {atr = 0:{1:}, atr = 1{1:,2:,3:}}
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
        diccionarioFinal = {}

        for elem in diccionarioSolucion.keys():
          diccionarioFinal[elem] = {}
          for clase in diccionarioSolucion[elem].keys():
            
            diccionarioFinal[elem][clase] = diccionarioSolucion[elem][clase] / conteoClase[clase] 
          
        i = 0
        
        print(diccionarioFinal)
        return diccionarioFinal
      else:
        pass



      pass

def dist_normal(m,v,n):
      if (v == 0):
        v += math.pow(10, -6)

      exp = -(math.pow((n-m), 2)/(2*v))
      base = 1/math.sqrt(2*math.pi*v)
      densidad = base*math.pow(math.e,exp)
      return densidad

 
    
  

##############################################################################


