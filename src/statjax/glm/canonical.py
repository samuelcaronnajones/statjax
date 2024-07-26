from . import GLM
from . links import *
from oryx.distributions import Normal,Bernoulli, Poisson, InverseGaussian
from . . probability import GammaNEF



class NormalGLM(GLM):
    def __init__(self, link = identity_link, **kwargs):
        super().__init__(link, Normal,  (1.,),  **kwargs)


def zero_intializer(X, *args, **kwargs):
    return jnp.zeros(X.shape[1])
class BernoulliGLM(GLM):
    def __init__(self, link  = logit_link, **kwargs):
        return super().__init__(link, Bernoulli, initializer =zero_intializer, **kwargs)


class PoissonGLM(GLM):
    def __init__(self, link = log_link, **kwargs):
        super().__init__(link, Poisson, **kwargs)



class GammaGLM(GLM):
    def __init__(self, link = inverse_link, **kwargs):
        return super().__init__(link, GammaNEF, (1.,), **kwargs)



class InverseNormalGLM(GLM):
    def __init__(self, link = inverse_squared_link, **kwargs):
        return super().__init__(link, InverseGaussian, (1.,), **kwargs)

