
from Datos import Datos
from Clasificador import *
from ClasificadorKNN import *
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('ConjuntosDatosP2/wdbc.csv')
clasificador = ClasificadorKNN()

validacionSimple = EstrategiaParticionado.ValidacionSimple(10,5)
K = 3
errores = []
for j in range(5):
    
    print(clasificador.validacion(validacionSimple,dataset,clasificador,seed=None, laPlace = False))

    #print(dataset.extraeDatos(validacionSimple.particiones[0].indicesTest))

    #clasificador.calcularMediaDesviacion(dataset.datos,dataset.nominalAtributos)
    #clasificador.normalizarDatos(dataset.datos,dataset.nominalAtributos)#TODO: normaliza es muy lento, hay que ver que está pasando. Además que si le pasas un porcentaje de la tabla no esta normalizando despues esos campos
    #print(f'Indices Test:\n{validacionSimple.particiones[0].indicesTest}')
    #print(f'Indices Test:\n{dataset.extraeDatos(validacionSimple.particiones[0].indicesTest)}')
#     for i in range(5):
        
#         predicciones = clasificador.clasifica(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest), dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain),dataset.nominalAtributos,K)
        
#         #print(f'{dataset.extraeDatos(validacionSimple.particiones[i].indicesTest)} vs {predicciones}')
#         error = clasificador.error(dataset.extraeDatos(validacionSimple.particiones[i].indicesTest),predicciones)

#         #print(error)
#         errores.append(error) 
# print(f'Media de los errores sin normalizar: {np.mean(errores) * 100}%')



# dataset.datos = clasificador.entrenamiento(dataset.datos, dataset.nominalAtributos)


# # dataset = Datos('ConjuntosDatosP2/wdbc.csv')
# clasificador = ClasificadorKNN()

