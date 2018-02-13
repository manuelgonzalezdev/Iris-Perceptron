import numpy as np

class AdalineGD (object):
    '''
    Attributes
        -   eta             => Learning rate [0.0,1.0]
        -   n_iter          => Number of iterations over dataset
        -   random_state    => Random seed for random weight
        -   _weights        => Vector of weights
        -   _cost           => Sum-of-squares cost function value in each iteration
    '''

    def __init__(self, eta = 0.01, n_iter= 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self._cost = []

    def fit(self, X, y):

        random_gen = np.random.RandomState(self.random_state)
        self._weights =  random_gen.normal(scale=0.01, size= X.shape[1] + 1)

        self._cost = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self._weights[0] += self.eta * errors.sum()
            self._weights[1:] += self.eta * X.T.dot(errors)
            cost = (errors**2).sum() * 0.5
            self._cost.append(cost)
        
        return self


    def predict(self, X):
        return np.where( self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def activation(self, X):
        return X

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]