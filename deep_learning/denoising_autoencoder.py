"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import numpy

import theano
import theano.tensor as Tensor
from theano.tensor.shared_randomstreams import RandomStreams


class DenoisingAutoencoder(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """


    def initialize_weights(self, numpy_rng, n_hidden, n_visible, W):
        """
        note : W' was written as `W_prime` and b' as `b_prime`

        W is initialized with `initial_W` which is uniformely sampled
        from -4*sqrt(6./(n_visible+n_hidden)) and
        4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
        converted using asarray to dtype
        theano.config.floatX so that the code is runable on GPU
        """

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX)
            W = theano.shared(
                value=initial_W,
                name='denoising_autoencoder_weights',
                borrow=True
            )
        self.W = W
        
        self.W_prime = self.W.T  # tied weights


    def initialize_biases(self, n_hidden, n_visible, bhid, bvis):
        """docstring for initialize_biases"""

        # b corresponds to the bias of the hidden
        self.b = (
            theano.shared(
                value=numpy.zeros(
                   n_hidden,
                   dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
            if not bhid
            else bhid
        )

        # b_prime corresponds to the bias of the visible
        self.b_prime = (
            theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            if not bvis
            else bvis
        )


    def __init__(self, numpy_rng, theano_rng=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        self.theano_rng = (
            RandomStreams(numpy_rng.randint(2 ** 30))
            if not theano_rng
            else theano_rng
        )

        self.initialize_weights(numpy_rng, n_hidden, n_visible, W)
        self.initialize_biases(n_hidden, n_visible, bhid, bvis)

        self.params = [self.W, self.b, self.b_prime]
        self.cost_fn = None

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return Tensor.nnet.sigmoid(Tensor.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  Tensor.nnet.sigmoid(Tensor.dot(hidden, self.W_prime) + self.b_prime)

    def cost(self, inputs, corruption_level):
        """docstring for cost"""
        if self.cost_fn == None:
            tilde_x = self.get_corrupted_input(inputs, corruption_level)
            y = self.get_hidden_values(tilde_x)
            z = self.get_reconstructed_input(y)
            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            L = - Tensor.sum(inputs * Tensor.log(z) + (1 - inputs) * Tensor.log(1 - z), axis=1)
            # note : L is now a vector, where each element is the
            #        cross-entropy cost of the reconstruction of the
            #        corresponding example of the minibatch. We need to
            #        compute the average of all these to get the cost of
            #        the minibatch
            self.cost_fn = Tensor.mean(L)
        return self.cost_fn
    
    def updates(self, inputs, learning_rate, corruption_level):
        """docstring for updates"""
        return [
            (param, param - learning_rate * gparam) 
            for param, gparam in zip(
                self.params,
                Tensor.grad(self.cost(inputs, corruption_level), self.params)
            )
        ]
