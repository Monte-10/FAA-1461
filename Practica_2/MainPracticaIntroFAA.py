
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntosDatosP2/wdbc.csv')
clasificador = ClasificadorKNN()


validacionSimple = EstrategiaParticionado.ValidacionSimple(10,5)
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(4)

validacionSimple.creaParticiones(dataset.datos)

errores = []
#print(dataset.extraeDatos(validacionSimple.particiones[0].indicesTest))

clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos
#print(f'Indices Test:\n{validacionSimple.particiones[0].indicesTest}')
#print(f'Indices Test:\n{dataset.extraeDatos(validacionSimple.particiones[0].indicesTest)}')
for i in range(5):
    
    predicciones = clasificador.clasifica(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest), dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain),dataset.nominalAtributos,3)
    
    #print(f'{dataset.extraeDatos(validacionSimple.particiones[i].indicesTest)} vs {predicciones}')
    error = clasificador.error(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest),predicciones)

    #print(error)
    errores.append(error) 
print(f'{np.mean(errores) * 100}%')