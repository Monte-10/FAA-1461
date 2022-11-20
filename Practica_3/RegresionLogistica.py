from Clasificador import Clasificador
from Datos import Datos
import numpy as np
import random
import math
import EstrategiaParticionado as EstrategiaParticionado
from ClasificadorKNN import ClasificadorKNN


class RegresionLogistica(Clasificador):

    def __init__(self) -> None:
        super().__init__()
        self.matrizConfusion = np.empty((2,2)) #¿asumo que los valores de clase siempre van a ser 0 y 1? De momento si. Inicializo con tam de matrix 2x2


    def getVectorAleatorio(self,nominalAtributos):
        lista = []
        for i in range(len(nominalAtributos)): #el vector tiene tamanio de x, es decir: sin incluir la clase
            lista.append(random.uniform(-0.5, 0.5))
        return np.array(lista)
    
    def sig(self,x):
        try:
            ret =  1/(1 + np.exp(-x))
            return ret
        except Exception:
            return 0
    
    def score(self,datosTest,prediccion):
        clases = datosTest.iloc[:,-1].to_numpy().astype('int64')
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        counter = 0
        for elem in clases:
            # print(f'{type(elem)}  ============== {type(prediccion[counter])}')
            # print(f'real: {elem}  ============== pred: {prediccion[counter]}')
            if elem == prediccion[counter]:
                if elem == 1:
                    # print("tp + 1\n")
                    tp += 1
                else:
                    # print("tn + 1\n")
                    tn += 1
            else:
                if elem == 1:
                    # print("fn + 1\n")
                    fn += 1 
                else:
                    # print("fp + 1\n")
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





    def entrenamiento(self,gradiente,num_epocas, datosTrain, nominalAtributos, diccionario):
        w = self.getVectorAleatorio(nominalAtributos)
        #print(f'w ---- {w}')
        n = gradiente
        n_epocas = num_epocas
        datos = datosTrain.iloc[:,:-1].to_numpy().astype('float64') #aunque parezcan raros, los esta cambiando bien.
        clases = datosTrain.iloc[:,-1].to_numpy().astype('int64')
        clases[clases != 1] = 0
        e = 0
        while(e < n_epocas):
            contador = 0
            for xj in datos:
                xj = np.append(1, xj)
                a = np.dot(w,xj)
                #print(f'w producto escalar xj {w} {xj} = {a}')
                
                posteriori = self.sig(a)
                #print(f"posteriori = {posteriori}")
                #print(f'posteriori de {a} =  {posteriori}')
                #print(f'La clase que equivale a esta fila del numpy es: {clases[contador]}')
                w = w - n*(posteriori - clases[contador])*xj
                #print(f' \n------------------->el nuevo valor de w es {w}')
                #print(posteriori)
                #una vez consiga parsear los datos a float, obtener w * xj
                contador += 1
                
            
            
            e+=1
        #para cada elemento de datosTrain
        return w

        #obtener w * x siendo x cada fila de la tabla.


        #actualizar el valor de w
    def clasificacion(self,w,datosTest):
        datos = datosTest.iloc[:,:-1].to_numpy().astype('float64')
        clases = datosTest.iloc[:,-1].to_numpy().astype('int64')
        clases[clases != 1] = 0

        clasesSolucion = []
        for xj in datos:
            xj = np.append(1, xj)
            #print(f'Xj: {xj} w:{w}')
            a = np.dot(w,xj)
            #print(f'\n\nPor lo tanto a es: {a}')
            if a > 0:
                clasesSolucion.append(1)
            else:
                clasesSolucion.append(0)

        clasificador = np.array(clasesSolucion)
        
        #print(clasificador)
        error = 0

        self.score(datosTest, clasificador)

        for i in range(len(clasificador)):
            #print(f'prediccion: {clasificador[i]} clase: {clases[i]}')
            if clasificador[i] == clases[i]:
                pass
            else:
                error +=1
        return error

        #TODO: ¿Que hace que una clase sea C1 o C2. Como digo que C1 es 1 y C2 es 0?
        #TODO: ¿Al hacer el np.dot(w,xj) w no debe ser (wo,w) y xj (x0,xj)?


    

    
if __name__ == '__main__':
    dataset = Datos('ConjuntosDatosP2/pima-indians-diabetes.csv')
    rl = RegresionLogistica()
    n_epocas = 10
    gradiente = 1
    errores = []

    # Si queremos normalizar:
    # dataset.datos = ClasificadorKNN().normalize(dataset.datos)

    for j in range(50):
        validacionSimple = EstrategiaParticionado.ValidacionSimple(25,5)
        validacionSimple.creaParticiones(dataset.datos)
        for i in range(5): 

            datosTrain = dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain)
            datosTest = dataset.extraeDatos(validacionSimple.particiones[i].indicesTest) 
            w = rl.entrenamiento(gradiente,n_epocas,datosTrain,dataset.nominalAtributos,dataset.diccionario)
            #print(f'W------------------------------------------------------{w}')
            error = rl.clasificacion(w,datosTest)
            errores.append(error/len(datosTest))
    print(f'El error para pima es: {np.mean(errores)}')
    print(f"Los valores de la matriz de confusion para este primer dataset es: \n{rl.matrizConfusion[0][0]}  {rl.matrizConfusion[0][1]}\n{rl.matrizConfusion[1][0]}  {rl.matrizConfusion[1][1]}")

    dataset = Datos('ConjuntosDatosP2/wdbc.csv')
    rl2 = RegresionLogistica()
    n_epocas = 10
    gradiente = 1
    errores = []

    #Si queremos normalizar:
    # dataset.datos = ClasificadorKNN().normalize(dataset.datos)
    
    for j in range(50):
        validacionSimple = EstrategiaParticionado.ValidacionSimple(25,5)
        validacionSimple.creaParticiones(dataset.datos)
        for i in range(5): 

            datosTrain = dataset.extraeDatos(validacionSimple.particiones[i].indicesTrain)
            datosTest = dataset.extraeDatos(validacionSimple.particiones[i].indicesTest) 
            w = rl2.entrenamiento(gradiente,n_epocas,datosTrain,dataset.nominalAtributos,dataset.diccionario)
            #print(f'W------------------------------------------------------{w}')
            error = rl2.clasificacion(w,datosTest)
            errores.append(error/len(datosTest))
    print(np.mean(errores))
    print(f'El error para wdbc es: {np.mean(errores)}')
    print(f"Los valores de la matriz de confusion para este primer dataset es: \n{rl2.matrizConfusion[0][0]}  {rl2.matrizConfusion[0][1]}\n{rl2.matrizConfusion[1][0]}  {rl2.matrizConfusion[1][1]}")
