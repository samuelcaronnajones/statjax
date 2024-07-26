from . . core import LinearModel, fit_r2, fit_r2_adj, se_from_cov, ci_from_z_se
from jax import grad, vmap, jit
from jax.lax import while_loop
from jax.scipy.linalg import solve

from oryx.core.ppl import random_variable, joint_log_prob
from oryx.core import inverse
from functools import partial
from typing import Callable

import jax.numpy as jnp
import optax


def glm_init_beta(X, y, link, inverse_link, error_distribution, aux_params):
    v = error_distribution(y, *aux_params).variance()

    eta = link(y)
    
    d_eta= vmap((grad(inverse_link)))(eta) 
    w = d_eta**2 / v

    
    beta_init = solve((X.T *w) @ X , (X.T *w ) @ eta)
    return beta_init

def glm_pass(aux_params, error_distribution, inverse_link , X, beta, key):
    mu = vmap(inverse_link)(X @ beta)
    y_i = lambda mu_i, key: random_variable((error_distribution(mu_i, *aux_params,)))(key)
    return random_variable(vmap(y_i, in_axes=(0,None)), name = "y")(mu, key)



def adam_fit_aux(loss, param, steps = 1000000):
    g = grad(loss)
    optimizer = optax.adam(1e-3)    
    initial_state = optimizer.init(param)
    loss_prev = jnp.inf

    def cond(arg):
        (i, loss_prev, param,_) = arg
        return (i < steps) & (jnp.abs(loss_prev - loss(param)) > 1e-6)
    @jit
    def step(arg):
        (i, loss_prev, param, opt_state) = arg
        loss_prev = loss(param)
        grads = g(param)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return (i + 1, loss_prev, param,opt_state)
    
    return while_loop(cond, step, (0, loss_prev, param,initial_state))


@jit
def enet_penalty(beta, lam, alpha):
    return 2 * lam * alpha * jnp.sum(jnp.abs(beta)) + lam * (1 - alpha) * jnp.sum(beta**2)



    
def glm_fit(X,y, link, inverse_link, error_distribution, aux_params, ctol = 1e-8, epochs=1000 , lam = 0, alpha = .5, initializer = glm_init_beta ):
    # guess initial beta using y as mu
    beta_init = initializer(X, y, link, inverse_link, error_distribution, aux_params)

    @jit
    def nll(beta, aux_params):
        mu = inverse_link(X @ beta)
        return -1 * jnp.sum(error_distribution(mu, *aux_params).log_prob(y ))
    
    # guess aux params : this is to remove dependence on starting value. 
    
    if len(aux_params) > 0:
        aux_params =  tuple([(link(y).std()) for _ in aux_params])


    # @jit
    def fisher_w(eta, mu, aux_params):  
        v = error_distribution(mu, *aux_params).variance()
        d_eta= vmap((grad(inverse_link)))(eta) 
        w = d_eta**2 / v

        return w
    # @jit
    def update_beta_ls(args):
        beta, i, history = args
        i += 1
        eta = X @ beta
        mu  = inverse_link(eta)
        z = eta + (y -mu) * vmap(grad(link))(mu)
        
        w = fisher_w(eta, mu, aux_params)
        beta_sgn = jnp.sign(beta)
        n = X.shape[0]
        beta = solve((X.T * w) @ X + (jnp.eye(X.shape[1]) *  lam*n * (1-alpha)+ 2 * n * lam * alpha * beta_sgn), (X.T * w) @ z)
        loss =nll(beta, aux_params) + enet_penalty(beta, lam, alpha)
        history = history.at[i].set(loss)
        return (beta, i, history)

    def cond_fn(args):
        _,  i, history = args
        return (i < epochs) & (jnp.abs(history[i - 1] - history[i]) > ctol)

    history = jnp.zeros(epochs)
    history = history.at[0].set(jnp.inf)
    beta, niter, loss_history=  while_loop(cond_fn, update_beta_ls, (beta_init , 0, history))

    converged  = ~cond_fn((beta, niter, loss_history))
    output_dict = {"beta": beta, "beta_train_iterations": niter,"beta_train_method": "irls", "train_losses": loss_history, "converged": converged}


    # fit aux with the new beta 
    if len(aux_params) > 0:
        i, _, aux_params, _ =  adam_fit_aux(jit(partial(nll, beta_init)),aux_params )
        output_dict.update({"aux_params_train_iterations": i, "aux_params_train_method": "adam"})

    output_dict["aux_params"] = aux_params

    output_dict["nll"] = nll(beta, aux_params)
    if len(aux_params) > 0:
        output_dict["nll_penalized"] = nll(beta, aux_params) + enet_penalty(beta, lam, alpha)

    return output_dict

def glm_score(model, X, y): 
    return -(joint_log_prob(partial(glm_pass,
                                    model.aux_params,
                                    model.error_distribution,
                                    model.link, 
                                    X,
                                    model.beta)
                                    ))({"y": y})
def glm_J(X,link, dist, beta, aux_params):
    eta = X @ beta
    def mu(eta):
        return inverse(link)(eta)
    
    v = dist(mu(eta), *aux_params).variance()
    d_eta= vmap((grad(mu)))(eta) 
    w = d_eta**2 / v

    return X.T * w @ X

def cov_glm(obj):
    return jnp.linalg.inv(glm_J(obj.X.values, obj.link, obj.error_distribution, obj.beta, obj.aux_params))

def glm_predict(model, X):
    return model.inverse_link(X @ model.beta)

def glm_aic(model):
        k = model.X.values.shape[1]
        return 2 * k + 2 * model.nll

def glm_bic(model):
        n, k  = model.X.values.shape
        return jnp.log(n)  * k + 2 * model.nll

class GLM(LinearModel):
    def __init__(self, link: Callable, error_distribution: random_variable, aux_params: tuple = (), lam = 0, alpha = .5, initializer = glm_init_beta, **kwargs):
        self.link = link
        self.inverse_link = inverse(link)
        self.error_distribution = error_distribution
        self.lam = lam
        self.alpha = alpha # aux params will be intiialized after fitting.
        self.initializer = initializer
        if not isinstance(aux_params, tuple):
            aux_params = (aux_params,)

        def fit(model, X, y):
            return glm_fit(X, y, model.link, model.inverse_link, model.error_distribution, aux_params, lam = model.lam, alpha = model.alpha,initializer = model.initializer)
   
        saved_stats = {"r2": fit_r2,
                        "r2_adj": fit_r2_adj,
                        "AIC": glm_aic,
                        "BIC": glm_bic}
        
        
        if lam == 0:
             saved_stats.update({"cov": cov_glm,
                                 "se": se_from_cov,
                                 "ci": ci_from_z_se})
           
        super().__init__(fit, glm_predict, glm_score,
                          saved_stats = saved_stats,
                          **kwargs)