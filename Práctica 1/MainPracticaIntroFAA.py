
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntoDatos/german.csv')
# print(dataset.datos)
# print(type(dataset.datos['Atr2']))
# print(dataset.datos['Atr2'].mean())

#print(dataset.datos["Class"][0]) #selecciona, de la columna Class, el valor que está en la fila 0
#print(dataset.datos.keys())
clasificador = ClasificadorNaiveBayes()
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(5)
validacionCruzada.creaParticiones(dataset.datos)
total = []
for i in range(5):
    print(validacionCruzada.particiones[i].indicesTrain)

clasificador.entrenamiento(dataset.extraeDatos(total) , dataset.nominalAtributos, laplace=True)
#clasificador.entrenamiento(dataset.extraeDatos(validacionSimple), dataset.nominalAtributos, laplace=False)
"""
estrategiaUno=EstrategiaParticionado.ValidacionCruzada(10)
estrategiaDos=EstrategiaParticionado.ValidacionSimple(50,10)
estrategiaUno.creaParticiones(dataset.datos)
#print(estrategiaUno)
estrategiaDos.creaParticiones(dataset.datos)
print(estrategiaDos)
"""
