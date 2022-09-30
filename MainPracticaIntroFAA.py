from Datos import Datos
import numpy as np
import EstrategiaParticionado
dataset = Datos('ConjuntoDatos/tic-tac-toe.csv')
print(dataset.datos)





dataset2 = Datos('ConjuntoDatos/german.csv')
print(dataset2.datos)


datos = np.random.random((10, 2))


vs = EstrategiaParticionado.ValidacionSimple(30, 3)
vs.creaParticiones(datos)

for i in range(3):
    print("\n\n")
    print("***Test***:", vs.particiones[i].indicesTest, "\n")
    print("***Train***:", vs.particiones[i].indicesTrain, "\n")
    
    
datos2 = np.random.random((8, 2)) 

vc = EstrategiaParticionado.ValidacionCruzada(4)
vc.creaParticiones(datos2)
      
for i in range(4):
    print("\n\n")
    print("***Test***:", vc.particiones[i].indicesTest, "\n")
    print("***Train***:", vc.particiones[i].indicesTrain, "\n")