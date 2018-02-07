import numpy as np

class Perceptron(object):
    '''
    Attributes
        -   eta             => Learning rate [0.0,1.0]
        -   n_iter          => Number of iterations over dataset
        -   random_state    => Random seed for random weight
        -   _weights        => Vector of weights
        -   _errors         => Number of misclassifications per iteration
    '''

    def __init__(self, eta = 0.01, n_iter= 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        random_gen = np.random.RandomState(self.random_state)
        self._weights =  random_gen.normal(scale=0.01, size= X.shape[1] + 1)
        self._errors = []
        print("Starting training")
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self._weights[0] += update
                self._weights[1:] += update * xi
                errors += 0 if update == 0 else 1
            self._errors.append(errors)
            print('Iteration {} - errors {}'.format(i, self._errors[-1]))

    def predict(self, X):
        return np.where( self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(self._weights[1:], X) + self._weights[0]

    