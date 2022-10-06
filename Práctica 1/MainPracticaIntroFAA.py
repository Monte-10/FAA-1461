from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntoDatos/tic-tac-toe.csv')
#print(dataset.datos["Class"][0]) #selecciona, de la columna Class, el valor que está en la fila 0
#print(dataset.datos.keys())
clasificador = ClasificadorNaiveBayes()
clasificador.entrenamiento(dataset.datos, dataset.nominalAtributos, None)
"""
estrategiaUno=EstrategiaParticionado.ValidacionCruzada(10)
estrategiaDos=EstrategiaParticionado.ValidacionSimple(50,10)

estrategiaUno.creaParticiones(dataset.datos)
#print(estrategiaUno)
estrategiaDos.creaParticiones(dataset.datos)
print(estrategiaDos)
"""
