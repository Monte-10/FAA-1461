from Clasificador import Clasificador
import numpy as np
class ClasificadorKNN(Clasificador):


  def __init__(self) -> None:
    super().__init__()
    self.matrizConfusion = np.empty((2,2)) #¿asumo que los valores de clase siempre van a ser 0 y 1? De momento si. Inicializo con tam de matrix 2x2
  #Se eligen estos metodos aqui porque(y además porque forma parte del entrenamiento de k-nn) para cumplir con la estructura de metodo que se nos da en el enunciado, es necesario itener variables de instancia que ugarden el valor de la desviacion tipica y la media para la normalizacion de los datos por lo tanto tendrá que ir dentro de cada objeto clasificador vecinos, yo creo que debería ser en datos porque -> porque todas las operaciones que se realicen sobre los datos, deben estar encapsuladas aqui. De tal forma que puedas operar sobre un dataset sin tener que involucrar a otras clases para reducir dependencias
  
  @staticmethod
  def normalize(copy):
    d = {}
    for elem in copy.keys():
      if elem != 'Class':
        d[elem] = 'float64'
    
    df = copy.astype(d, copy = True)
    
    
    df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    
    return df
    
    

  def entrenamiento(self,datosTotales, atributosNominal): #entrenamiento tiene que tener los mismos valores que la clase abstracta que implementa.
    
    return ClasificadorKNN.normalize(datosTotales)
    



  def clasifica(self, datosTest, datosTrain, nominalAtributos,k):
    atr = datosTest.keys()
    #print(f'Los datosTest tienen las siguientes columnas:{atr} cuya len es {len(atr)}')
    atributos = [] #distancia independiente entre atributos, luego habra que sumar todas estas distancias
    
    listaClases = []
    for index1,x1 in datosTest.iterrows():
      distancias = [] #distancia total de cada fila de test, con todas las de train (index, valor). Cada fila de test, tendra este array. 

      for index2,x2 in datosTrain.iterrows():
        lista1 = []
        lista2 = []
        for i in x1.items(): #datos
          if i[0] != atr[-1]: #la distancia no la calculamos con la clase. Sino con el resto de los atributos
            if checkfloat(i[1]):
              lista1.append(float(i[1]))
            else:
              lista1.append(int(i[1]))
        for i in x2.items(): #clusters
          if i[0] != atr[-1]:
            if checkfloat(i[1]):
              lista2.append(float(i[1]))
            else:
              lista2.append(int(i[1]))
        #print(f'\nINDICE:{index1}\n{lista1}\nINDICE:{index2}\n{lista2}\n\n' )
        dist = distance(lista1,lista2)
        #print(f'Distancia:{dist}')
        #print(f'la distancia euclidiana es: {sum(atributos)}')
        distancias.append((index2,dist))
        
        

      listaOrdenada = []
      listaOrdenada = sorted(distancias,key=lambda x: x[1])
      #print(listaOrdenada)
      #print(f'{listaOrdenada}')
      #print(f'Los k minimo valor de todas las distancias son:{listaOrdenada[:k]}')
      
      repeated_values = []
      for j in range(k):
        # print(f'vamos a meter el valor {datosTrain.loc[clasesSolucion[j][0]]} a la lista porque es uno de los menores')
        repeated_values.append(datosTrain[atr[-1]].loc[listaOrdenada[j][0]])    #seleccionaremos los k valores mas repetidos
        #print(f'{index1} Localizame el atributo en la posicion {listaOrdenada[j][0]} de la clase en train. Yo creo que es -> {datosTrain[atr[-1]].loc[listaOrdenada[j][0]]}')
      #listaClases.append((index1,max(set(repeated_values),key=repeated_values.count))) #aniadimos a la lista el indice de test con la clase mas predicha
      #print(f'{x1} -> predice la clase {max(set(repeated_values),key=repeated_values.count)}')
      listaClases.append(max(set(repeated_values),key=repeated_values.count)) #aniadimos a la lista el indice de test con la clase mas predicha
      #print(f'Seleccionamos los k valores mas repetidos{repeated_values}')
      #print(f'El valor mas repetido para el datosTest {index1} es:{max(set(repeated_values),key=repeated_values.count)}')  
      
      
      
    # print(f'clases solucion {listaClases}')
    self.clasesPredichas = listaClases
    #print(f'{listaClases} es la lista que contiene la clase más predicha. Supuestamente deberia estar ordenada por el mismo orden que indices Test. De tal forma que la posicion 0 de esta lista equivale al primer indice de indicestest ')
    self.score(datosTest,listaClases)
    return listaClases

  def score(self,datosTest,prediccion):
    clases = datosTest.iloc[:,-1].to_numpy().astype('int64')
    prediccion[prediccion != 1] = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    counter = 0
    for elem in clases:
        # print(f'{type(elem)}  ============== {type(prediccion[counter])}')
        # print(f'real: {elem}  ============== pred: {prediccion[counter]}')
        # print(type(elem),type(predicciones[counter]))
        if elem == int(prediccion[counter]):
            if elem == 1:
                #print("tp + 1\n")
                tp += 1
            else:
                #print("tn + 1\n")
                tn += 1
        else:
            if elem == 1:
                #print("fn + 1\n")
                fn += 1 
            else:
                #print("fp + 1\n")
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


def checkfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


            
def distance(list1,list2):
    """Euclidean Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    #print(f'La distancia entre {list1} y {list2} es: {sum(squares) ** .5}')
    return sum(squares) ** .5
    # distancia = 0
    # total = len(list1) - 1
    # for i in range(total):
    #     distancia +=  np.linalg.norm(list1[i]-list2[i])
    # print(f'La distancia entre {list1} y {list2} es: {distancia}')
    # return distancia
    

def euclideanDistance(x1, x2):
  if isinstance(x1, str):
    if checkfloat(x1) is True:
      x1 = float(x1)
    else:
      x1 = int(x1)

  if isinstance(x2, str):
    if checkfloat(x2) is True:
      x2 = float(x2)
    else:
      x2 = int(x2)
    
  return np.linalg.norm(x1-x2)

 