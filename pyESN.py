import numpy as np

class ESN():

    def __init__(self, n_inputs: int, n_outputs: int, n_reservoir: int = 500,
                 input_scale=1, feedback_scale=1, spectral_radius=0.95,
                 teacher_forcing: bool = True, sparsity=0, noise=0.001,
                 bias=0.01, ridge=10**-10, rng=np.random.default_rng()):
        """
        An implementation of Echo State Network.
        The specification of the network mainly follows Lu et al (2017), while
        the leakage rate is fixed to be zero.
        See https://aip.scitation.org/doi/10.1063/1.4979665 for more details.

        :param n_inputs: number of input dimensions
        :param n_outputs: number of output (teacher) dimensions
        :param n_reservoir: number of reservoir nodes
        :param input_scale: scale of input weights
        :param feedback_scale: scale of feedback weights
        :param spectral_radius: spectral radius of the recurrent weight matrix
        :param teacher_forcing: whether to feed the output (teacher) back to the network
        :param sparsity: proportion of recurrent weights set to zero
        :param noise: scale of noise in the network dynamics
        :param bias: bias constant in activation function
        :param ridge: ridge regression parameter
        :param rng: random generator
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.input_scale = input_scale
        self.feedback_scale = feedback_scale
        self.spectral_radius = spectral_radius
        self.teacher_forcing = teacher_forcing
        self.sparsity = sparsity
        self.noise = noise
        self.bias = bias
        self.ridge = ridge
        self.rng = rng

        self._initweights()

    def _initweights(self):
        """
        Initialize the adjacency matrix of the reservior network and the input weight matrix
        """
        # the adjacency matrix, beginning with a random matrix in range [-1,1):
        A = self.rng.random((self.n_reservoir, self.n_reservoir)) - 0.5
        # delete some connections to satisfy the average degree:
        A[self.rng.random(A.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(A)))
        # rescale them to reach the requested spectral radius:
        self.A = A * (self.spectral_radius / radius)

        # generate a random input weight matrix:
        self.W_in = (self.rng.random((self.n_reservoir, self.n_inputs
                                      )) * 2 - 1)*self.input_scale
        # generate a random feedback weight matrix:
        if self.teacher_forcing:
            self.W_feedb = (self.rng.random((self.n_reservoir, self.n_outputs
                                             )) * 2 - 1)*self.feedback_scale
        return

    def _update(self, current_state, input_pattern, teacher_pattern):
        """
        performs one update step.
        i.e., computes the next network state by applying the adjacency matrix
        to the last state and the input/feedback weight matrix to an input/teacher
        """
        preactivation = (np.dot(self.A, current_state)
                         + np.dot(self.W_in, input_pattern))+self.bias
        if self.teacher_forcing:
            preactivation += np.dot(self.W_feedb, teacher_pattern)
        return (np.tanh(preactivation)
                + self.noise * (self.rng.random(self.n_reservoir) - 0.5))

    def fit(self, inputs, teachers):
        """
        Collect the network's reaction to training data, training output weights.

        :param inputs: array of dimensions (steps * n_inputs)
        :param teacher: array of dimension (steps * n_outputs)
        """
        # detect and correct possible errors:
        if len(teachers) != (steps := len(inputs)):
            raise ValueError("teacher and input do not match")
        if inputs.ndim < 2:
            inputs = np.expand_dims(inputs, 1)
        if inputs.shape[1] != self.n_inputs:
            raise ValueError("incorrect input dimension")
        if teachers.ndim < 2:
            teachers = np.expand_dims(teachers, 1)
        if teachers.shape[1] != self.n_outputs:
            raise ValueError("incorrect teacher dimension")

        # pre-allocate memory for network states:
        states = np.zeros((steps, self.n_reservoir))
        # let the network evolve according to inputs:
        for n in range(steps-1):
            states[n+1] = self._update(states[n], inputs[n+1], teachers[n])
        # remember the last state for later:
        self.laststate = states[-1]
        self.lastoutput = teachers[-1]

        # disregard the first few states:
        transient = min(int(steps / 10), 300)
        states = states[transient:]
        teachers = teachers[transient:]
        # learn the weights, i.e. solve output layer quantities W_out and c
        # that make the reservoir output approximate the teacher sequence:
        states_mean = np.mean(states, axis=0)
        teachers_mean = np.mean(teachers, axis=0)
        states_delta = states-states_mean
        teachers_delta = teachers-teachers_mean
        Id = np.eye(self.n_reservoir)
        self.W_out = teachers_delta.T.dot(states_delta).dot(
            np.linalg.inv((states_delta.T).dot(states_delta)+self.ridge*Id))
        self.c = teachers_mean-self.W_out.dot(states_mean)
        self.measure_error(states, teachers)
        return

    def measure_error(self, states, teachers):
        outputs = np.squeeze(np.dot(states, self.W_out.T))+self.c
        self.mse = np.mean((outputs-teachers)**2)

    def predict(self, inputs):
        """
        Apply the learned weights to the network's reactions to new input.

        :param inputs: array of dimensions (N * n_inputs)
        :return: array of network output signal
        """
        steps = len(inputs)
        # detect and correct possible errors:
        if inputs.ndim < 2:
            inputs = np.expand_dims(inputs, 1)
        if inputs.shape[1] != self.n_inputs:
            raise ValueError("incorrect input dimension")

        # pre-allocate memory for outputs
        outputs = np.vstack(
            [self.lastoutput, np.zeros((steps, self.n_outputs))])
        state = self.laststate

        # let the network evolve according to inputs:
        for n in range(steps):
            state = self._update(state, inputs[n], outputs[n])
            outputs[n+1] = np.dot(self.W_out, state)+self.c

        return outputs[1:]
    
