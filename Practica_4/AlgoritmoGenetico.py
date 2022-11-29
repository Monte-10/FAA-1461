import numpy as np
import random
import matplotlib.pyplot as plt
from Clasificador import *
from Datos import Datos

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

    def fitness(self, datosTrain, reglas):
        # Para cada regla se activan las que cumplan la condicion contra el dataset
        # De las que cumplen se coge lo que predice la mayoria y se compara
        # Si acierta +1 y si falla pos nah
        # Para compararlo lo ideal es hacer un AND bit a bit, para eso habra que hacer OHE del dataset
