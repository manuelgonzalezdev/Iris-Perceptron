import numpy as np

class AdalineGD (object):
    ''' ADAptative LInear NEuron classifier '''
    ''' Gradient Descent '''


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

class AdalineSGD (object):
    ''' ADAptative LInear NEuron classifier '''
    ''' Stochastic Gradient Descent '''


    '''
    Attributes
        -   eta             => Learning rate [0.0,1.0]
        -   n_iter          => Number of iterations over dataset
        -   random_state    => Random seed for random weight
        -   shuffle         => Shuffle training data each iteration if True
        -   _weights        => Vector of weights
        -   _cost           => Sum-of-squares cost function value in each iteration
    '''

    def __init__(self, eta = 0.01, n_iter= 50, shuffle = True, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):

        self.initialize_weights(X.shape[1])
        self._cost = []

        # random_gen = np.random.RandomState(self.random_state)
        # self._weights =  random_gen.normal(scale=0.01, size= X.shape[1] + 1)

        # self._cost = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self.shuffle_data(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self._cost.append(avg_cost)
        return self


            # net_input = self.net_input(X)
            # output = self.activation(net_input)
            # errors = y - output
            # self._weights[0] += self.eta * errors.sum()
            # self._weights[1:] += self.eta * X.T.dot(errors)
            # cost = (errors**2).sum() * 0.5
            # self._cost.append(cost)
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(X, y)
        return self

    def shuffle_data(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self._weights = self.rgen.normal(loc=0.0, scale=0.01, size = 1 + m)
        self.w_initialized = True

    def update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self._weights[1:] += self.eta * xi.dot(error)
        self._weights[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def predict(self, X):
        return np.where( self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def activation(self, X):
        return X

