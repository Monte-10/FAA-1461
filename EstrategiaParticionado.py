from abc import ABCMeta,abstractmethod
import random
import numpy as np

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

  def __str__(self):
    return "Indices entrenamiento:" + str(self.indicesTrain) + "\nIndices Test " + str(self.indicesTest)
    

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  def __init__(self):
    self.particiones = []
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass

  def __str__(self) -> str:
    cadena = ""
    for eleme in self.particiones:
      cadena += str(eleme) + "\n"
    return cadena
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)

  #@args pTest porcentaje que se coge para el test
  #      nEjec numero de ejecuciones
  def __init__(self, pTest, nEjec):
      super().__init__()
      self.pTest = pTest
      self.nEjec = nEjec
  
  def creaParticiones(self,datos,seed=None):
    random.seed(seed)
    self.particiones = []
    longitudDatos = np.shape(datos)[0]
    longitudTest = int((self.pTest/100)*longitudDatos)

    valores = [i for i in range(longitudDatos)]

    for i in range(self.nEjec):
      self.particiones.append(Particion())

      random.shuffle(valores)

      self.particiones.append(Particion())

      random.shuffle(valores)

      self.particiones[-1].indicesTest = valores[:longitudTest]
      self.particiones[-1].indicesTrain = valores[longitudTest:]
  
  
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  def __init__(self, numeroParticiones):
    super().__init__()
    self.numeroParticiones = numeroParticiones
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    self.particiones = []
    longitudDatos = np.shape(datos)[0]
    longitudPorcion = int(longitudDatos/self.numeroParticiones)

    valores = [i for i in range(longitudDatos)]
    random.shuffle(valores)

    for i in range(self.numeroParticiones):
      self.particiones.append(Particion())

      #Calculo indices
      fromTest = i*longitudPorcion
      toTest = fromTest + longitudPorcion

      #Asigno indices
      self.particiones[-1].indicesTest = valores[fromTest:toTest]
      self.particiones[-1].indicesTrain = [i for i in valores if i not in self.particiones[-1].indicesTest]
    
    
