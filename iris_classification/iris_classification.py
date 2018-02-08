import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron

IRIS_DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def plot_feature_distribution(features, labels):
    colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange']
    markers = ['o', 'x', 'v', 'H', '+', 's']
    for i, feature in enumerate(features):
        plt.scatter(feature[:,0], feature[:,1], color=colors[i % len(colors)], marker=markers[i % len(markers)], label = labels[i])
    plt.legend(loc='upper left')
    plt.show()

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    colors = ['red', 'blue', 'yellow', 'green', 'orange']
    markers = ['o', 'x', 'v', 'H', '+']
    colorMap = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    x_array = np.array([xx1.ravel(), xx2.ravel()])
    x_array = x_array.T
    Z = [classifier.predict(x) for x in x_array]
    Z = np.array(Z)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = colorMap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for i, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[i],
                    marker=markers[i],
                    label=cl,
                    edgecolors='black')

    plt.title('Decision regions')
    plt.legend(loc='upper left')
    plt.show()

def plot_training_results(classifier):
    plt.plot(range(1, classifier.n_iter + 1), classifier._errors, marker='o')
    plt.title('Training session')
    plt.xlabel('Iteration')
    plt.ylabel('Errors')
    plt.show()

df = pd.read_csv(IRIS_DATASET_URL, header=None)

# select setoca and versicolor
Y = df.iloc[0:100, 4].values
Y = np.where(Y =='Iris-setosa', 1, -1)

# extract sepal and petal length
X = df.iloc[0:100, [0, 2]].values

# See features distribution
'''
features = []
features.append(X[:50])
features.append(X[50:100])
labels = ['setosa', 'versicolor']
plot_feature_distribution(features, labels)
'''

ppn = Perceptron(eta =0.1, n_iter=10)
ppn.fit(X,Y)

plot_training_results(ppn)
plot_decision_regions(X,Y, classifier=ppn)

