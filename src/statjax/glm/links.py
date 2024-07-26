import jax.numpy as jnp
from oryx.bijectors import Sigmoid
from oryx.core import custom_inverse
from jax.scipy.special import erfinv, ndtr

ve = 1e-8


def identity_link (x):
    return x

def log_link(mu):
    return jnp.log(mu)

@custom_inverse
def logit_link(mu):
    mu = jnp.clip(mu,ve, 1-ve)
    return Sigmoid().inverse(mu)

logit_link.def_inverse_unary(lambda eta: Sigmoid()(eta), f_ildj=lambda x: 0)


def probit_link_inverse(eta):
    return ndtr(eta)
@custom_inverse
def probit_link(mu):
    mu = jnp.clip(mu,ve, 1-ve)
    return jnp.sqrt(2) * erfinv(2*mu - 1)

probit_link.def_inverse_unary(probit_link_inverse, f_ildj=lambda mu:  (-jnp.log(2 * jnp.pi) - mu**2) / 2.)

def inverse_link (mu):
    return  -1/mu

def inverse_squared_link(mu):
    return  -mu **(-2) /2