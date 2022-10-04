# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np
import csv
import pandas as pd

class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        self.nominalAtributos = []
        self.datos = {}
        self.diccionario = {}
        file = open(nombreFichero)
        csvReader = csv.reader(file)          
        self.getOrderedDict(csvReader)
        
    def getHeaders(self,csvReader):
        headers = next(csvReader)    
        return headers
        
            

    def getOrderedDict(self,csvReader):
        headers = self.getHeaders(csvReader)
        rows = []
        for row in csvReader:
            rows.append(row)
        for h in rows[0]:
            if(h.isnumeric()):
                self.nominalAtributos.append(False)
            elif isinstance(h, str):
                self.nominalAtributos.append(True)
            else:
                raise ValueError

        counter = 0
        primero = []
        
        for h in headers:
            
            primero.append([])

            for fila in rows:
                primero[counter].append(fila[counter])
                ##TODO: la insercion de los datos no es directa, sino que es el valor ordenador alfabeticamente
            counter += 1        
        
        counter = 0
        d = {}
        for elem in headers:
            d[elem] = primero[counter]
            counter += 1
        
        #CREA el dataFrame a partir de los datos
        self.datos = pd.DataFrame(d)
        
        counter = 0
        for h in headers:
            self.diccionario[h] = self.getDict(h,self.nominalAtributos[counter])
            counter += 1    
        self.datos.replace(self.diccionario,inplace = True) 
        print(self.datos)
        
    def getDict(self,header,bandera):
        secuencia = 1
        sample_set = set(self.datos[header]) 
        sample_set = sorted(sample_set)
        temp = {}
        if(bandera == True):
            for elem in sample_set:
                temp[elem] = secuencia
                secuencia += 1
        
        return temp
    
    
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass

