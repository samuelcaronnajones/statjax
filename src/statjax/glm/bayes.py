import jax.random as random
import jax.numpy as jnp
import blackjax

from oryx.core.ppl import random_variable, joint_log_prob
from oryx.core import inverse

from jax import vmap, jit
from jax.lax import scan
from functools import partial

from . glm  import glm_init_beta, glm_predict
from .. core import LinearModel, fit_r2, fit_r2_adj

def BayesGLM_pass(aux_params, error_distribution, prior, inverse_link, X_shape, key):
    x_key, beta_key, error_key = random.split(key, 3)
    X = random_variable(lambda key: jnp.ones(X_shape), name = "X")(x_key)
    beta = random_variable(prior, name = "beta")(beta_key)

    eta_i = X @ beta
    mu = vmap(inverse_link)(eta_i)
    
    y_i = lambda mu_i, error_key: random_variable((error_distribution(mu_i, *aux_params,)))(error_key)
    y= random_variable(vmap(y_i, in_axes=(0,None)), name = "y")(mu, error_key)
    return y



def BayesGLM_score(obj, X, y):
    draw = partial(BayesGLM_pass, obj.aux_params, obj.error_distribution, obj.prior, obj.inverse_link, X.shape)
    
    density =  lambda params : - joint_log_prob(draw)(params)
    return  density({"X": X, "beta": obj.beta, "y": y})


def bayesGLM_mcmc_fit(X,y, link, error_distribution, prior, aux_params,  n_burnin=1000,n_samples=1000, rng_key = random.PRNGKey(0), initializer = glm_init_beta):
    
    inverse_link = inverse(link)
    beta_init = initializer(X, y, link, inverse_link, error_distribution,aux_params )
  
    if len(aux_params) > 0:
        aux_params =  tuple([(link(y).std()) for _ in aux_params])

    def mcmc_density(beta, aux_params):
        draw_base = partial(BayesGLM_pass, aux_params, error_distribution, prior, inverse_link, X.shape)
        return joint_log_prob(draw_base)({"X": X, "beta": beta, "y": y,})
    
    logdensity = lambda params : jit(mcmc_density)(**params)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = random.split(rng_key, num_samples)
        _, states = scan(one_step, initial_state, keys)

        return states

    initial_position = {"beta": beta_init, "aux_params": aux_params}
    rng_key, sample_key , warmup_key = random.split(rng_key, 3)

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=n_burnin)

    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, n_samples)
    mcmc_samples = states.position
    beta_point = jnp.percentile(mcmc_samples["beta"], 50,axis=0)

    aux = []
    for a in mcmc_samples["aux_params"]:
        aux.append( jnp.percentile(a, 50,axis=0))
    aux = tuple(aux)
        
    return {"beta": beta_point,
            "aux_params": aux,
            "mcmc_samples": mcmc_samples,
            }

def BayesGLM_ci(obj):
    posterior = obj.mcmc_samples["beta"]

    ci = jnp.percentile(posterior, jnp.array([2.5, 97.5]), axis=0)
    return ci 

class BayesGLM(LinearModel):
    def __init__(self, link, error_distribution, beta_prior_distribution, aux_params = (), n_burnin=1000,n_samples=5000, rng_key = random.PRNGKey(0), **kwargs):

        def fit(model, X, y):
            return bayesGLM_mcmc_fit(X, y, model.link, model.error_distribution, model.prior, aux_params, n_burnin=n_burnin, n_samples=n_samples, rng_key=rng_key)
        
        self.prior = beta_prior_distribution
        self.error_distribution = error_distribution
        self.link = link
        self.inverse_link = inverse(link)

        super().__init__(fit, glm_predict, BayesGLM_score,{"nlppd": lambda obj: -BayesGLM_score(obj, obj.X, obj.y), #posterior point density
                                                           "ci": BayesGLM_ci,
                                                              "r2": fit_r2,
                                                                "r2_adj": fit_r2_adj}, **kwargs)
                                                        