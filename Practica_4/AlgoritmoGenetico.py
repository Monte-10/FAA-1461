import numpy as np
import random
import matplotlib.pyplot as plt
from Clasificador import *
from Datos import Datos
import sklearn.preprocessing as pre
import EstrategiaParticionado as EstrategiaParticionado
import math


class AG(Clasificador):
    def __init__(self, prob_mutacion, n_generaciones, max_reglas, prob_combinacion, tam_poblacion):
        self.n_generaciones = n_generaciones
        self.max_reglas = max_reglas
        self.prob_mutacion = prob_mutacion
        self.prob_combinacion = prob_combinacion
        self.tam_poblacion = tam_poblacion

    def generaPoblacion(self, dataset,diccionario):
        total = 0
        for clave in diccionario.keys():
            if clave != "Class":
                #print(f'El tamanio de las keys de {clave} es:{len(diccionario[clave].keys())}')
                total += len(diccionario[clave].keys())
        total += 1 #reservamos un bit para la clase
        self.tam_regla = total
        tam_array = total * self.max_reglas
        # print(f'Total = {total} * {self.max_reglas}')
        # print(tam_array)
        return np.random.randint(2, size=tam_array) #el individuo contiene tambien la clase

    def intraReglas(self, lista_1, lista_2):
        
        lista1 = lista_1.tolist()
        lista2 = lista_2.tolist()
        #print(f'litsa1: {len(lista1)} , lista2: {(lista2)}')
        indice = random.randint(0,len(lista1))
        
        listaSolucion1 = lista1[0:indice]
        listaSolucion1 += lista2[indice:]
        listaSolucion2 = lista2[0:indice]
        listaSolucion2 += lista1[indice:]

        #print(len(listaSolucion1))
        #print(len(listaSolucion2))
        return np.array(listaSolucion1), np.array(listaSolucion2)
    
    def mutacion(self, lista):
        for indice in range(len(lista)):
            num = random.random()
            if num < self.prob_mutacion:
                lista[indice] = 1 - lista[indice]     
        
        return lista


    def encodeAsOneHot(datosTrain):
        #CUIDADO: Se hace el onehotencode con todo los atributos del dataset incluida la clase. Por lo que las reglas se codifican con 2 bits y nuestro dataset con 1. A tener en cuenta.
        #CUIDADO: clase 1 -> 0 clase 2 -> 1 en bits. Para los dos datasets
        enc = pre.OneHotEncoder()
        enc.fit(datosTrain)
        trans = enc.transform(datosTrain).toarray().astype(int)
        # print(f'trans antes:')
        # print(trans)
        trans = np.delete(trans,-2,1) #to avoid having two bits for class
        # print(f'trans despues:')
        # print(trans)

        return trans

    def compareDataToRule(self,data, rules):
        clases = []
        acierto = 0
        error = 0


        for dato in data:
            flagReglas = False #si no entra ninguna regla, se cuenta como fallo.
            aciertoReglas = 0
            falloReglas = 0
            for rule in rules:
                

                #andTotal = np.zeros(len(rule)).astype(int)
                andTotal = []
                for i in range(len(dato)-1):
                    #np.append(andTotal,rule[i] and data[i]) 
                    andy = dato[i] and rule[i]
                    andTotal.append(andy)
                andTotal.append(rule[-1])    
                # print(f'regla {rule} andy : {andTotal} dato {dato}')
                comparison = np.array(andTotal[:-1]) == dato[:-1]
                if comparison.all():
                    #print(f'La regla {andTotal[:-1]} se activa para el dato {dato[:-1]}')
                    flagReglas=True
                    #En este punto, la regla se activa y entra a comparar el valor de la clase. Si acierto +1, si fallo error +1
                    if andTotal[-1] == dato[-1]:
                        # print(f'ACIERTO')
                        aciertoReglas +=1 
                    else:
                        # print(f'FALLO')
                        falloReglas +=1

            if flagReglas is True: #es decir, al menos una regla a entrado a valorar la clase
                if aciertoReglas > falloReglas:
                    acierto += 1
                else:
                    error += 1
            else:
                error += 1
        #print(f'Acierto = {acierto}, Error = {error}')
        return acierto,error
                
    
    def fitness(self, datos, individuo):
        reglas = []
        lista = []
        i = 0
        while i<len(individuo):#lo unico que hacemos aqui es elaborar el individuo como una regla de reglas de tal forma que luego podremos trabajar de ofrma independiente con cada regla
            lista = individuo[i:i+self.tam_regla]
            reglas.append(lista)
            i += (self.tam_regla)
       
        acierto,error = self.compareDataToRule(datos,reglas)
        #print(f'Vamos a estudiar el score de las reglas: {reglas} acierto:{acierto/(acierto+error)} error: {error/(acierto+error)}')
        return acierto/(acierto+error)


        # Para cada regla se activan las que cumplan la condicion contra el dataset
        # De las que cumplen se coge lo que predice la mayoria y se compara
        # Si acierta +1 y si falla pos nah
        # Para compararlo lo ideal es hacer un AND bit a bit, para eso habra que hacer OHE del dataset
        
        

    #     atributo1: X O                       data:  X  O  E    Z
    #     XO                                   fila1: 01 01 1000 001
    #     10                                   regla: 11 10 1001 111
    #                                           AND   11 11 1111 111
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

    def roulette_wheel_selection(self,population):
        population_fitness = sum(population)
        chromosome_probabilities =  [chromosome/population_fitness for chromosome in population]
        return np.random.choice(population, p=chromosome_probabilities)

    def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
        # Creacion de la poblacion inicial (# individuos) con el esquema propuesto: 
        # reglas de longitud fija y número de reglas variable por individuo
        descendientes = []
        flagPrimerosIndividuos = False
        for n in range(self.n_generaciones):
            individuos = descendientes
            descendientes = []
            #print("entro")
            if flagPrimerosIndividuos is False:
                for i in range(self.tam_poblacion):
                    individuos.append(self.generaPoblacion(datosTrain,diccionario))
                    flagPrimerosIndividuos= True
            # print(f"Los progenitores son : {individuos}")
            trans = AG.encodeAsOneHot(datosTrain)
            fitness_individuo = []
            for individuo in individuos:
               
                fitness_individuo.append(self.fitness(trans,individuo))
            
            for i in range(math.floor(self.tam_poblacion/2)):
                seleccion1 = self.roulette_wheel_selection(fitness_individuo)
                index1 = fitness_individuo.index(seleccion1)
                seleccion2 = self.roulette_wheel_selection(fitness_individuo)
                index2 = fitness_individuo.index(seleccion2)
                coin = random.random()
                if coin < self.prob_combinacion:
                    descendiente1, descendiente2 = self.intraReglas(individuos[index1],individuos[index2])
                else:
                    descendiente1, descendiente2 = individuos[index1],individuos[index2]

                mut1 = self.mutacion(descendiente1)
                mut2 = self.mutacion(descendiente2)

                if all(el==1 for el in mut1) | all(el==0 for el in mut1):
                    if all(e==1 for e in descendiente1) | all(e==0 for e in descendiente1):
                        descendientes.append(individuos[index1])        #si tras mutar son todos 1's o 0's y tras combinar tambien lo fueron, nos quedamos con el progenitor
                        descendientes.append(individuos[index2])
                    else:
                        descendientes.append(descendiente1)        #si tras mutar son todos 1's pero tras combinar no lo son, nos quedamos con la combinacion.
                        descendientes.append(descendiente2)
                else:
                    descendientes.append(mut1)  
                    descendientes.append(mut2)
            # print(f'Los descendientes son : {descendientes}')

        #tras todas las generaciones, nos quedamos con el mejor descendiente
        fitneses = []
        for descendiente in descendientes:
            fitneses.append(self.fitness(trans,descendiente))

        mejor = fitneses.index(max(fitneses))

        # print(f'El mejor individuo tras train es: {descendientes[mejor]}')
        # print(f'Y tiene un score de: {fitneses[mejor]}')
        # print(trans)
        return descendientes[mejor]
        
        
        # print(f'Los fitness son {fitness_individuo}')
        # print(f'El indice es: {index}')
        # print(f'Tras la ruleta rusa el individuo seleccionado es: {fitness_individuo[index]}')


        
        ###ruleta rusa###





        #calculamos los vastagos


        # Evolución de la población, calculando el fitness de cada
        # individuo y aplicando operadores genéticos (y elitismo si se
        # considera) para obtener la nueva población durante el
        # número de generaciones establecido
        
        # El fitness se calcula a partir de los datos de train
        
        # Al final de la evolución se tiene un individuo (conjunto de
        # reglas) con el mejor fitness (aciertos en train) posible
        

    def clasifica(self,datosTest,individuo,diccionario=None):
        trans = AG.encodeAsOneHot(datosTest)
        
        # print(type(datosTest))
        # print(type(individuo))
        acierto = self.fitness(trans,individuo)
        print(f'El acierto de nuestro clasificador es del: {acierto*100}% por lo que el error es el: {(1- acierto)*100}%')
        # Se evalúa el mejor individuo evolucionado frente a
        # los datos de test. Se predicen las clases en función de las
        # reglas activadas y las clases correspondientes a esas
        # activaciones de forma similar al voto por mayoría de K-nn
        
        

if __name__=='__main__':
    dataset = Datos('ConjuntosDatosP4/titanic.csv')
    algoritmoGenetico = AG(0.01,100,5,0.85,100) #penultimo parametro -> proporcion de arrastrar el mejor
    
    validacionSimple = EstrategiaParticionado.ValidacionSimple(30,1) #reservamos poco para el train de forma temporal para debuggear
    validacionSimple.creaParticiones(dataset.datos)
    
    datosTrain = dataset.extraeDatos(validacionSimple.particiones[0].indicesTrain)
    datosTest = dataset.extraeDatos(validacionSimple.particiones[0].indicesTest)

    
    
    # print(trans)
    # print(f'trans {trans[0]}')
    # print(f'trans {trans[1]}')
    # print(f'trans {trans[2]}')
    
    mejor = algoritmoGenetico.entrenamiento(datosTrain,{},dataset.diccionario)
    # print(f'El mejor es: {mejor} y lo puedes probar con el dataset')
    #mejor = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    algoritmoGenetico.clasifica(datosTest,mejor)

    print(datosTrain)

    