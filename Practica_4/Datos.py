# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
from Clasificador import *
import numpy as np
import csv
import pandas as pd

class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        self.nominalAtributos = [] #true si es discreto, false si numerico
        self.datos = {} #{atrN{:[]}}
        self.diccionario = {}
        file = open(nombreFichero)
        csvReader = csv.reader(file)          
        self.getOrderedDict(csvReader)

     
    @staticmethod
    def normalize(copy):
        d = {}
        for elem in copy.keys():
            if elem != 'Class':
                d[elem] = 'float64'

        df = copy.astype(d, copy = True)


        df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

        return df
   
    def getHeaders(self,csvReader):
        headers = next(csvReader)    
        return headers
    
    
            

    def getOrderedDict(self,csvReader):
        headers = self.getHeaders(csvReader)

        rows = []
        for row in csvReader:
            rows.append(row)
        for h in rows[0]:
            if(h.isnumeric() or isfloat(h)):
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
                if len(fila) > 0:
                    primero[counter].append(fila[counter])
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
    
    
    # Devuelve el subconjunto de los datos cuyos indices se pasan como argumento
    def extraeDatos(self,idx):
        return np.take(self.datos,idx,axis=0)



def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


