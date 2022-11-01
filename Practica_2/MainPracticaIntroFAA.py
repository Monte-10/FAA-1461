
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntosDatosP2/pima-indians-diabetes.csv')
clasificador = ClasificadorKNN()


validacionSimple = EstrategiaParticionado.ValidacionSimple(40,5)
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(4)

validacionSimple.creaParticiones(dataset.datos)
errores = []
#print(dataset.extraeDatos(validacionSimple.particiones[0].indicesTest))

clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos

print(f'Indices Test:\n{dataset.extraeDatos(validacionSimple.particiones[0].indicesTest)}')
for i in range(1):
    
    predicciones = clasificador.clasifica(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest), dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain),dataset.nominalAtributos,3)
    print(predicciones)
    error = clasificador.error(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest),predicciones)
    print(error)
    errores.append(error) 
print(f'{np.mean(errores) * 100}%')