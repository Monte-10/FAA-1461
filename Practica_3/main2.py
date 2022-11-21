
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('/mnt/c/Users/alexm/Documents/FAA-1461/Practica_3/ConjuntosDatosP2/pima-indians-diabetes.csv')
# print(dataset.datos)
# print(type(dataset.datos['Atr2']))
# print(dataset.datos['Atr2'].mean())

#print(dataset.datos["Class"][0]) #selecciona, de la columna Class, el valor que está en la fila 0
#print(dataset.datos.keys())
clasificador = ClasificadorNaiveBayes()

validacionSimple = EstrategiaParticionado.ValidacionSimple(10,3)
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(4)
# validacionSimple.creaParticiones(dataset.datos)
# total = []
# print(dataset.diccionario)
#clasificador.entrenamiento(dataset.extraeDatos(validacionSimple.particiones[0].indicesTrain) , dataset.nominalAtributos, False,dataset.diccionario)
#clasificador.entrenamiento(dataset.extraeDatos(validacionSimple), dataset.nominalAtributos, laplace=False)



error = []

error += clasificador.validacion(validacionCruzada, dataset, clasificador)


print(f"Los valores de la matriz de confusion para este primer dataset es: \n{clasificador.matrizConfusion[0][0]}  {clasificador.matrizConfusion[0][1]}\n{clasificador.matrizConfusion[1][0]}  {clasificador.matrizConfusion[1][1]}")

# error = []
# for i in range(20):
#     error += clasificador.validacion(validacionSimple, dataset, clasificador)
# print(statistics.mean(error))
# print(statistics.stdev(error))


# error = []
# for i in range(20):
#     error += clasificador.validacion(validacionCruzada, dataset, clasificador, laPlace = True)
# print(statistics.mean(error))
# print(statistics.stdev(error))

# error = []
# for i in range(20):
#     error += clasificador.validacion(validacionSimple, dataset, clasificador, laPlace = True)
# print(statistics.mean(error))
# print(statistics.stdev(error))

#clasificador.validacion(validacionSimple, dataset, clasificador, laPlace = True)