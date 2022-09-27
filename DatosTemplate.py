# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np
import csv
import pandas as pd

class Datos:



    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        #TODO: settear los valores correctamente a nominalAtributos : [lista de booleanos que indica, para cada header si es nominal o numerico]
        self.nominalAtributos = False
        #datos = pd.DataFrame(d) donde d es un {"columnName1":[data],"columnName2":[data]}
        self.datos = {}
        self.diccionario = None
        #obtengo el objeto file
        file = open(nombreFichero)
        csvReader = csv.reader(file)  
        
        self.getOrderedDict(csvReader)
        #llamada a getRowByPosOrdered -> devuelve un diccionario ordenado alfabeticamente con clave el atributo de entrada, y valor una secuencia numerica
        
        
        #self.getOrderedDict(csvReader)    
            
    
    
    
    
    
    
    
    def getHeaders(self,csvReader):
        headers = next(csvReader)
        return headers
        
            

    def getOrderedDict(self,csvReader):
        headers = self.getHeaders(csvReader)
        
        rows = []
        for row in csvReader:
            rows.append(row)
        counter = 0
        primero = []
        
        for h in headers:
            primero.append([])
            for fila in rows:
                primero[counter].append(fila[counter])
                ##TODO: la insercion de los datos no es directa, sino que es el valor ordenador alfabeticamente
            counter += 1        
        
        #llegado a este punto tengo una lista por header. tengo que contruir ahora el dataframe de forma que {"header"  : lista[header]}
        counter = 0
        d = {}
        for elem in headers:
            d[elem] = primero[counter]
            counter += 1
        
        self.datos = pd.DataFrame(d)
        print(self.datos)
        print(self.prueba(headers))
        '''for elem in headers:
            
            self.datos[headers] = [item[counter] for item in rows]
            counter += 1
            print(rows[:][counter])'''


    #llamar a este metodo por cada header, y obtener el diccionario. Una vez obtenido, crear 
    def getDict(self,headers, index):
        secuencia = 0
        sample_set = set(self.datos[headers[i]]) 
        print(sample_set)
        temp = {}
        for elem in sample_set:
            temp[elem] = secuencia
            secuencia += 1
        return temp
        
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass


if __name__ == "__main__":

    datos = Datos("csv/german.csv")