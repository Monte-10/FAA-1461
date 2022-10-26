import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
#import seaborn as sb
 
#%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from abc import ABCMeta,abstractmethod
import numpy as np
from Datos import Datos

dataset = pd.read_csv('ConjuntosDatosP2/pima-indians-diabetes.csv')
dataset.head(10)
'''print(dataset.describe())
print(dataset.hist())'''

X = dataset[['Pregs','Plas','Pres','Skin','Test','Mass','Pedi','Age']].values
y= dataset[['Class']].values

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

vecinos = 7
knn = KNeighborsClassifier(vecinos)
knn.fit(X_train, y_train)
print('Precisión de K-NN en train: {:.2f}'.format(knn.score(X_train,y_train)))
print('Precisión de K-NN en test: {:.2f}'.format(knn.score(X_test,y_test)))

#MATRIZ DE CONFUSION PARA DETALLAR ACIERTOS Y FALLOS
pred = knn.predict(X_test)
print('\nMatriz:')
print(confusion_matrix(y_test, pred))
print('\nClasificación:')
print(classification_report(y_test, pred))

h = .02

# Crea mapas de colores
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])

# Creamos instancia de KNN para meter los datos
weights='distance'
clf = KNeighborsClassifier(vecinos, weights='distance')
clf.fit(X, y)

# Traza el límite de la decisión. Para ello, asignaremos un color a cada punto de la malla [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Poner el resultado en su color
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Poner el entrenamiento
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    
patch0 = mpatches.Patch(color='#FF0000', label='1')
patch1 = mpatches.Patch(color='#ff9933', label='2')
patch2 = mpatches.Patch(color='#FFFF00', label='3')
patch3 = mpatches.Patch(color='#00ffff', label='4')
patch4 = mpatches.Patch(color='#00FF00', label='5')
plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])

    
plt.title("Clasificacion de clase: (k = %i, weights = '%s')"  % (vecinos, weights))

plt.show()