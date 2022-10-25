
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntosDatosP2/wdbc.csv')
clasificador = ClasificadorKNN()

# print(dataset.datos)
# print(type(dataset.datos['Atr2']))
# print(dataset.datos['Atr2'].mean())

#print(dataset.datos["Class"][0]) #selecciona, de la columna Class, el valor que está en la fila 0
#print(dataset.datos.keys())


validacionSimple = EstrategiaParticionado.ValidacionSimple(30,1)
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(4)

validacionSimple.creaParticiones(dataset.datos)
clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos
predicciones = clasificador.clasifica(dataset.extraeDatos(validacionSimple.particiones[0].indicesTest), dataset.extraeDatos(validacionSimple.particiones[0].indicesTrain),dataset.nominalAtributos,3)
errores = clasificador.error(dataset.extraeDatos(validacionSimple.particiones[0].indicesTest),predicciones)
print(f'{errores * 100}%')