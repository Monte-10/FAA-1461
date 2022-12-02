import numpy as np
import random
import matplotlib.pyplot as plt
from Clasificador import *
from Datos import Datos
import sklearn.preprocessing as pre
import EstrategiaParticionado as EstrategiaParticionado


class AG():
    def __init__(self, prob_mutacion, n_individuos, n_generaciones, max_reglas, max_seleccionados, tam_poblacion):
        self.n_individuos = n_individuos
        self.n_generaciones = n_generaciones
        self.max_reglas = max_reglas
        self.prob_mutacion = prob_mutacion
        self.max_seleccionados = max_seleccionados
        self.tam_poblacion = tam_poblacion

    def generaPoblacion(self, dataset, max_reglas):
        for clave in dataset.diccionario.keys(): # Tenemos que mirar si la clase la esta cogiendo o no
            print(clave)
            total += len(clave.keys())

        return np.zeros(total, max_reglas)

    def intraReglas(self, lista1, lista2):
        indice = random.randint(0,len(lista1))
        listaSolucion1 = lista1[0:indice]+lista2[indice+1:-1]
        listaSolucion2 = lista2[0:indice]+lista1[indice+1:-1]

        return listaSolucion1, listaSolucion2
    
    def mutacion(self, lista):
        for indice in range(lista):
            num = random.random(1)
            if num > self.prob_mutacion:
                lista[indice] = 1 - lista[indice]     
        
        return lista


    def encodeAsOneHot(datosTrain):
        enc = pre.OneHotEncoder()
        enc.fit(datosTrain)
        return enc.transform(datosTrain).toarray().astype(int)

    def compareDataToRule(data, rules):
        clases = []
        acierto = 0
        for rule in rules:
            andTotal = np.zeros(len(rule)).astype(int)
            for i in range(len(rule)):
                np.append(andTotal,rule[i] and data[i]) 
            print(andTotal)
            print(data)
            comparison = andTotal == data
            if comparison.all():
                print("es igual")
                
    def evaluaReglas(reglas, datos):
        # Reglas es una lista de listas, donde cada lista es una regla
        # Datos es una lista de listas, donde cada lista es un dato
        # Devuelve una lista de 1 y 0, donde 1 es que la regla coincide con el dato y 0 es que no coincide
        lista = []
        for regla in reglas:
            for dato in datos:
                if (regla[0:-1] == dato[0:-1]).all():
                    if regla[-1] == dato[-1]:
                        lista.append(1)
                    else:
                        lista.append(0)
        return lista
        
            
    # def dataComparationWithRules(data,rules): 
    #     clases = []
    #     # Comparacion de data y rules con puerta AND
    #     for i in range(len(data)):
    #         for j in range(len(rules)):
    #             if (np.logical_and(rules[j], data[i]) ) == data[i]:
    #                 print(f'{rules[j]}{data[i]}')
    #                 #clases.append(rules[j][-1])


    # def fitness(self, datosTrain, reglas):
    #     # Para cada regla se activan las que cumplan la condicion contra el dataset
    #     # De las que cumplen se coge lo que predice la mayoria y se compara
    #     # Si acierta +1 y si falla pos nah
    #     # Para compararlo lo ideal es hacer un AND bit a bit, para eso habra que hacer OHE del dataset

    #     atributo1: X O                       data:  X  O  E    Z
    #     XO                                   fila1: 01 01 1000 001
    #     10                                   regla: 11 10 1001 111
    #                                                 11 11 1111 111
    #                                                 ______________
    #                                                 01 00 1000 001
    #                                                 01 01 1000 001
    



    #     Pclass(2),Sex(2),Age(3),Class(2)
    #     2         male      3     1
    #     10        01      100     01
    #     11        10      001     10
    #     (pclass = 1 or pclass=2) and (sex=male) and (age = 1) then class = 2
    #     (pclass = 1 or pclass=2) and (sex=male) and (age = 3) then class = 2 fallo +1 
    #     (pclass = 1 or pclass=2) and (sex=male) and (age = 3) then class = 1 acierto +1

if __name__=='__main__':
    dataset = Datos('/mnt/c/Users/Gervasia y Gervasio/Documents/GitHub/FAA-1461/Practica_4/ConjuntosDatosP4/xor.csv')
    
    validacionSimple = EstrategiaParticionado.ValidacionSimple(10,1)
    validacionSimple.creaParticiones(dataset.datos)

    datosTrain = dataset.extraeDatos(validacionSimple.particiones[0].indicesTrain)
    
    trans = AG.encodeAsOneHot(datosTrain)
    print(f'trans {trans[0]}')
    AG.evaluaReglas(trans[0],np.array([[0,1,0,1,0,1],[1,1,1,1,1,1]]))