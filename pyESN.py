import numpy as np


class ESN():

    def __init__(self, n_inputs, n_reservoir=1000,
                 spectral_radius=0.95, average_degree=20,
                 input_scale=0.05, bias=0.01, ridge=10 ** (-10)):
        """
        An implementation of Echo State Network.
        The specification of the network mainly follows Lu et al (2017), while
        the leakage rate is fixed to be zero.
        See https://aip.scitation.org/doi/10.1063/1.4979665 for more details.

        :param n_inputs: number of input dimensions
        :param n_reservoir: number of reservoir nodes
        :param spectral_radius: spectral radius of the recurrent weight matrix
        :param average_degree: average degree of network nodes
        :param input_scale: scale of input weights
        :param bias: bias constant in activation function
        :param ridge: ridge regression parameter in Tikhonov regularization
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.average_degree = average_degree
        self.input_scale = input_scale
        self.bias = bias
        self.ridge = ridge
        self.initweights()

    def initweights(self):
        """
        Initialize the adjacency matrix of the reservior network and the input weight matrix
        """
        # create a random generator
        rng = np.random.default_rng()
        # the adjacency matrix, beginning with a random matrix in range [-1,1):
        A = rng.random((self.n_reservoir, self.n_reservoir))*2-1
        # delete some connections to satisfy the average degree:
        A[rng.random(A.shape) > self.average_degree/self.n_reservoir] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(A)))
        # rescale them to reach the requested spectral radius:
        self.A = A * (self.spectral_radius / radius)

        # generate a random input weight matrix:
        self.W_in = rng.random((self.n_reservoir, self.n_inputs)) * \
            (2*self.input_scale) - self.input_scale
        return

    def _update(self, current_state, input_pattern):
        """
        performs one update step.
        i.e., computes the next network state by applying the adjacency matrix
        to the last state and the input weight matrix to an input
        """
        input_pattern = np.array(
            input_pattern)  # this may avoid error when the input is a scalar
        preactivation = (self.A.dot(current_state)
                         + (self.W_in).dot(input_pattern)+self.bias)
        return np.tanh(preactivation)

    def fit(self, inputs, teacher):
        """
        Collect the network's reaction to training data, training output weights.

        :param inputs: array of dimensions (steps * n_inputs)
        :param teacher: array of dimension (steps x n_outputs)
        """
        # detect possible errors:
        if len(teacher) != len(inputs):
            raise ValueError("number of teacher and input do not match")
        if inputs[0].size != self.n_inputs:
            raise ValueError("incorrect input dimension")
        self.n_outputs = teacher[0].size  # number of output dimensions
        steps = len(inputs)
        # pre-allocate memory for network states:
        states = np.zeros((steps+1, self.n_reservoir))
        # let the network evolve according to inputs:
        for n in range(steps):
            states[n+1] = self._update(states[n], inputs[n])
        self.laststate = states[-1]
        # disregard the first few states:
        transient = min(int(steps / 10), 500)
        states = states[transient+1:]
        teacher = teacher[transient:]
        # learn the weights, i.e. solve output layer quantities W_out and c
        # that make the reservoir output approximate the teacher sequence:
        states_average = np.average(states, axis=0)
        teacher_average = np.average(teacher, axis=0)
        delta_states = states-states_average
        delta_teacher = teacher-teacher_average
        Id = np.identity(self.n_reservoir)
        self.W_out = (delta_teacher.T).dot(delta_states).dot(
            np.linalg.pinv(delta_states.T.dot(delta_states)+self.ridge*Id))
        self.c = teacher_average-self.W_out.dot(states_average)
        return

    def predict(self, inputs):
        """
        Apply the learned weights to the network's reactions to new input.

        :param inputs: array of dimensions (N * n_inputs)
        :return: array of network output signal
        """
        # detect possible errors:
        if inputs[0].size != self.n_inputs:
            raise ValueError("incorrect input dimension")
        steps = len(inputs)
        # pre-allocate memory for outputs
        outputs = np.zeros((steps, self.n_outputs))
        state = self.laststate
        # let the network evolve according to inputs:
        for n in range(steps):
            state = self._update(state, inputs[n])
            outputs[n] = np.dot(self.W_out, state)+self.c
        return outputs
