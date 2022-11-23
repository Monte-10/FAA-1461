from Clasificador import *
from ClasificadorKNN import *
from Datos import Datos
from Clasificador import *
import EstrategiaParticionado as EstrategiaParticionado
import RegresionLogistica as rl
import numpy as np
import matplotlib.pyplot as plt

from Datos import Datos
dataset = Datos('/mnt/c/Users/alexm/Documents/FAA-1461/Practica_3/ConjuntosDatosP2/pima-indians-diabetes.csv')

clasificador1 = ClasificadorNaiveBayes()

validacionSimple = EstrategiaParticionado.ValidacionSimple(10,3)
validacionCruzada = EstrategiaParticionado.ValidacionCruzada(4)

error = []

error += clasificador1.validacion(validacionCruzada, dataset, clasificador1)

print(f"Los valores de la matriz de confusion para este primer dataset es: \n{clasificador1.matrizConfusion[0][0]}  {clasificador1.matrizConfusion[0][1]}\n{clasificador1.matrizConfusion[1][0]}  {clasificador1.matrizConfusion[1][1]}")
