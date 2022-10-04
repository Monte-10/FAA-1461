from Datos import Datos
import EstrategiaParticionado as EstrategiaParticionado
dataset = Datos('csv/tic-tac-toe.csv')
estrategiaUno=EstrategiaParticionado.ValidacionCruzada(10)
estrategiaDos=EstrategiaParticionado.ValidacionSimple(50,10)

estrategiaUno.creaParticiones(dataset.datos)
#print(estrategiaUno)
estrategiaDos.creaParticiones(dataset.datos)
print(estrategiaDos)

