import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# select setoca and versicolor
Y = df.iloc[0:100, 4].values
Y = np.where(Y =='Iris-setosa', 1, -1)

# extract sepal and petal length
X = df.iloc[0:100, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label = 'setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label = 'versicolor')
# plt.xlabel('sepal length in cm')
# plt.ylabel('petal length in cm')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta =0.1, n_iter=10)
ppn.fit(X,Y)