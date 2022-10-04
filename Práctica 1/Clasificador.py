from abc import ABCMeta,abstractmethod
import numpy as np
import EstrategiaParticionado
from scipy.stats import norm

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
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
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    '''particionado.creaParticiones(dataset.datos)
    mErr = 0
    mErrCLP = 0
    for particion in particionado.particiones:
      datosTest = dataset.datos[particion.indicesTest, :]
      datosTrain = dataset.datos[particion.indicesTrain, :]
      
      self.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionario)
      res, resCLP = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionario)
      mErr += clasificador.error(datosTest, res)
      mErrCLP += clasificador.error(datosTest, resCLP)
    lTest = len(particionado.particiones)
    return mErr/lTest, mErrCLP/lTest'''
    
    
 
class ClasificadorNaiveBayes(Clasificador): 
    dicc_atrib = {}
    dicc_clas = {}
    
    def entrenamiento():
      """Entrenamiento"""
      
    def clasifica(self,datosTest,atributos,dicc):
      '''post = {}
      pri = {}
      lista = []
      bayes = []
      
      n = sum(list(self.dicc_clas.values()))
      
      for i, j in self.dicc_clas.items():
        pri.update({k:(j/n)})
        
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
        
        bayes.append((crear funcion reduce)reduce(lambda x, y: x*y, lista)*pri[i])
        post[cont][i] = bayes'''
      
      
    
  

##############################################################################

