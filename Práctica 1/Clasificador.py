from abc import ABCMeta,abstractmethod
from xmlrpc.client import boolean
import numpy as np
import EstrategiaParticionado
from scipy.stats import norm
from Datos import Datos

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
      probCondicionadas["at1"] = self.getProbabilidadesCondicionadas(prioris, datosTrain[datosTrain.keys()[i]], nominalAtributos[i],datosTrain[datosTrain.keys()[-1]],conteoClase) #TODO: falta aniadir al diccionario por cada atributo del excel
      i += 1
      
      
      #print("\n\n\nCondicionadas: " + str(probCondicionadas))



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
    '''
    def getProbabilidadesCondicionadas(self, prioris, datosTrain, nominalAtributo,clase,conteoClase): 
      diccionario = {}
      counter = 0
      valoresClase = []
      for valor in clase:
        if not valoresClase.__contains__(valor):
          valoresClase.append(valor)
        
      
      #diccionarioFinal -> {atr:{1:}}
      if(nominalAtributo is True):
        # for elem in datosTrain:
        #   for valor in valoresClase
        #   if diccionario.__contains__(elem):
           
        #     diccionario[elem] += 1
        #   else:
        #     diccionario[elem] = 1
           
        diccionarioSolucion = {}
        i = 0
               
        for elemClase in valoresClase:
          
          for elem in datosTrain: #diccionario con el conteo de todos los valores de la columna
            if( clase[i] == elemClase):              
              print("ENTRO")
              print(diccionarioSolucion)
              if diccionarioSolucion.__contains__(elem) and diccionarioSolucion[elem].__contains__(elemClase):
                print("contiene, sumo 1 a " + str(elem) + "|" + str(elemClase)  )
                diccionarioSolucion[elem][elemClase] += 1
              else:
                diccionarioSolucion[elem] = {elemClase: 1} 
            
            i += 1
          print(diccionarioSolucion)
          diccionarioFinal = {}

          for elem in diccionarioSolucion.keys():
            for clase in diccionarioSolucion[elem].keys():
              diccionarioFinal[elem] = {clase: diccionarioSolucion[elem][clase] / conteoClase[clase] }
            
          i = 0
          #TODO: gestionar de forma correcta los nombres porque esta haciendo la division por una clase que no le corresponde
          print(conteoClase)
          print(diccionarioFinal)
        return diccionarioFinal
      else:
        pass



      pass

 
    
  

##############################################################################


