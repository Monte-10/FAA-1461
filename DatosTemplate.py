# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np
import csv
import pandas as pd

class Datos:

    nominalAtributos = False
    datos = {} #esto debe usar pandas para crear el {:{}}
    diccionario = None

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        #obtengo el objeto file
        file = open(nombreFichero)
        csvReader = csv.reader(file)  
        
        self.getHeaders(csvReader)

        #llamada a getRowByPosOrdered -> devuelve un diccionario ordenado alfabeticamente con clave el atributo de entrada, y valor una secuencia numerica
        self.getOrderedDict(csvReader)    
            
    
    
    
    
    
    
    
    def getHeaders(self,csvReader):
        list = []
        for row in csvReader:
            list = row
            break

        for elem in list:
            self.datos[elem] = {}

    def getOrderedDict(self,csvReader):
        #list = []
        print(csvReader[self.datos[0]])




        
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass

if __name__ == "__main__":

    datos = Datos("csv/german.csv")