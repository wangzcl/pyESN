import torch


class multiESN():
    """
    A batch processing version of Echo State Networks
    """

    def __init__(self, n_network: int, n_reservoir: int, n_inputs: int, n_outputs: int,
                 input_scale=0.1, feedback=0, spectral_radius=1, sparsity=0,
                 noise=0, bias=0.02, ridge=1e-10) -> None:
        """
        :param n_network: number of networks
        :param n_reservoir: number of neurons in each network
        :param n_inputs: number of input dimensions
        :param n_outputs: number of output (teacher) dimensions
        :param input_scale: scale of input weights
        :param feedback: scale of feedback weights
        :param spectral_radius: spectral radius of the recurrent weight matrix
        :param teacher_forcing: whether to feed the output (teacher) back to the network
        :param sparsity: proportion of recurrent weights set to zero
        :param noise: scale of noise in the network dynamics
        :param bias: bias constant in activation function
        :param ridge: ridge regression parameter
        """
        print("initializing...")
        self.n_network = n_network
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.input_scale = input_scale
        self.feedback = feedback
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.bias = bias
        self.ridge = ridge
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initweights()
        print("initialization finished")
        return

    def _initweights(self) -> None:
        """
        Initialize the random adjacency matrices,input and feedback weights of each network
        Pre-allocate memory for output weights and last outputs
        """
        # the adjacency matrices, beginning with a random tensor in range [-0.5,0.5):
        A = torch.rand(self.n_network, self.n_reservoir,
                       self.n_reservoir, device=self.device) - 0.5
        # delete some connections to satisfy the average degree:
        A[torch.rand(*A.shape, device=self.device)
          < self.sparsity] = 0
        # compute the spectral radii of these weights:
        radius = torch.linalg.eigvals(A).abs().max(dim=1, keepdim=True).values
        # rescale them to reach the requested spectral radius:
        A *= ((self.spectral_radius / radius).unsqueeze_(1))
        self.A = A.cpu()
        del A

        # generate a random input weight matrix:
        self.W_in = ((torch.rand(self.n_network, self.n_reservoir,
                     self.n_inputs, device=self.device) * 2 - 1)*self.input_scale).cpu()
        # generate a random feedback weight matrix:
        if self.feedback:
            self.W_feedb = (self.feedback * (torch.rand(self.n_network, self.n_reservoir,
                                                        self.n_outputs, device=self.device) * 2 - 1)).cpu()
        # initialize the output coefficients W_out and c
        self.W_out = torch.empty(
            self.n_network, self.n_outputs, self.n_reservoir)
        self.c = torch.empty(self.n_network, self.n_outputs, 1)
        # pre-allocate memory for the last output/teacher of the training period
        self.laststate = torch.empty(self.n_network, self.n_reservoir)
        return

    def _reshape(self, *args) -> torch.Tensor:
        """
        Pretreat data before inputting them to the network
        :param *args: each element is a tuple of the form (array_like data, dim)
        """
        ans = (torch.tensor(
            arg[0], dtype=torch.float32).reshape(-1, arg[1]).to(self.device) for arg in args)
        return tuple(ans)

    def _update(self, current_state, input_pattern, teacher_pattern, A, W_in, W_feedb):
        """
        performs one update step.
        i.e., computes the next network state by applying the adjacency matrix A
        to the last state and the input/feedback weight matrix to an input/teacher
        """
        preactivation = (A @ current_state.unsqueeze(2)
                         ).squeeze_(2) + W_in @ input_pattern + self.bias
        if self.feedback:
            preactivation += (W_feedb @ teacher_pattern.unsqueeze_(-1)).squeeze_(-1)
        new_state = torch.tanh(preactivation)
        if self.noise:
            new_state += (2*self.noise) * (torch.rand(self.n_network,
                                                      self.n_reservoir, device=self.device) - 0.5)
        return new_state

    def fit(self, inputs, teachers, n_chosen: int, n_batch: int, transient: int = None) -> None:
        """
        Collect the networks's reaction to training data, training output weights.

        :param inputs: array_like data of the dimension (steps * n_inputs)
        :param teacher: array_like data of the dimension (steps * n_outputs)
        :param n_chosen: number of networks that performed best and to be chosen
        :param n_batch: split the multiESN in to n_batch pieces, dealing with one at a time to prevent memory error
        *if memory runs out, increase n_batch
        :param transient: number of steps at first to ignore
        """
        print("training...")
        # preprocess input data:
        inputs, teachers_raw = self._reshape(
            (inputs, self.n_inputs), (teachers, self.n_outputs))
        if (steps := len(teachers)) != (m := len(inputs)):
            raise ValueError(
                "length of teachers {} and inputs {} do not match".format(steps, m))
        # remember the last output for later prediction:
        self.lastoutput = teachers_raw[-1]
        # an identity matrix
        Id = torch.eye(self.n_reservoir, device=self.device)
        # pre-allocate memory for the training errors
        mse = torch.empty(self.n_network)
        # decide the transient number
        if transient is None:
            transient = min(int(steps / 10), 300)
        # split the multi-networks into chunks
        if self.n_network % n_batch:
            raise ValueError(
                "cannot split {} network into {} equal-size batches".format(self.n_network, n_batch))
        else:
            batch_size = self.n_network // n_batch
        A = self.A.split(batch_size)
        W_in = self.W_in.split(batch_size)
        if self.feedback:
            W_feedb = self.W_feedb.split(batch_size)

        # start batch processing
        for i in range(n_batch):
            # put a part of the multiESN into GPU
            A_temp = A[i].to(self.device)
            W_in_temp = W_in[i].to(self.device)
            W_feedb_temp = W_feedb[i].to(
                self.device) if self.feedback else None
            # pre-allocate memory for network states:
            states = torch.empty(batch_size, steps,
                                 self.n_reservoir, device=self.device)
            states[:, 0] = 0
            # let the network evolve according to inputs:
            for n in range(steps-1):
                states[:, n+1] = self._update(states[:, n], inputs[n+1],
                                              teachers_raw[n], A_temp, W_in_temp, W_feedb_temp)
            # disregard the first few states, and other necessary calculations:
            teachers = teachers_raw[transient:]
            states = states[:, transient:]
            # learn the weights, i.e. solve output layer quantities W_out and c
            # that make the reservoir output approximate the teacher sequence:
            teachers_mean = torch.mean(teachers, dim=0)
            teachers -= teachers_mean
            states_mean = torch.mean(states, dim=1)
            states -= states_mean.unsqueeze(1)

            W_out = teachers.T @ states @ torch.linalg.inv(
                (states.transpose(1, 2)@states+Id*self.ridge))
            c = teachers_mean.unsqueeze(
                1)-W_out @ states_mean.unsqueeze(2)

            states += states_mean.unsqueeze(1)
            teachers += teachers_mean
            # the real network output
            outputs = (states @ W_out.transpose(1, 2) + c)

            # save the results
            self.laststate[i*batch_size:(i+1)*batch_size] = states[:, -1]
            self.W_out[i*batch_size:(i+1)*batch_size] = W_out
            self.c[i*batch_size:(i+1)*batch_size] = c
            mse[i*batch_size:(i+1)*batch_size] = torch.mean((outputs -
                                                             teachers)**2, dim=(1, 2))
            # delete unuseful data to release memory
            del A_temp, W_in_temp, W_feedb_temp, states, states_mean, W_out, c, outputs
            torch.cuda.empty_cache()

        # choose the best of the networks
        self.n_chosen = n_chosen
        self.chosen_idx = torch.argsort(mse)[:n_chosen]
        print("train finished")
        return

    def predict(self, inputs) -> torch.Tensor:
        """
        Apply the learned weights to the network's reactions to new input.

        :param inputs: array_like data of the dimension (N * n_inputs)
        :return: network output signal
        """
        inputs = self._reshape((inputs, self.n_inputs))[0]
        steps = len(inputs)
        # pick out the chosen networks
        state = self.laststate[self.chosen_idx].to(self.device)
        A = self.A[self.chosen_idx].to(self.device)
        W_in = self.W_in[self.chosen_idx].to(self.device)
        W_feedb = self.W_feedb[self.chosen_idx].to(
            self.device) if self.feedback else None
        W_out = self.W_out[self.chosen_idx].to(self.device)
        c = self.c[self.chosen_idx].to(self.device)
        # pre-allocate memory for outputs
        outputs = torch.hstack((self.lastoutput.broadcast_to(
            (self.n_chosen, 1, self.n_outputs)), torch.empty(self.n_chosen, steps, self.n_outputs, device=self.device)))
        # let the network evolve according to inputs:
        for n in range(steps):
            state = self._update(
                state, inputs[n], outputs[:, n], A, W_in, W_feedb)
            outputs[:, n+1] = (W_out @ state.unsqueeze(2) + c).squeeze(2)
        # the average of all chosen network outputs is the final result
        ans = outputs[:, 1:].mean(dim=0).squeeze().cpu()
        return ans
