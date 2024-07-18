from functools import partial
from jax.lax import while_loop
from jax import value_and_grad, grad, jit, config

from optax import adam, apply_updates
from .core import LinearModel, cov_filler, predict_ols

import jax.numpy as jnp


config.update("jax_enable_x64", True)
ve = 1e-8


def nlm_fit(X, y, predict, loss_function, regularization= lambda x: 0,
                learning_rate=.1, epochs=10000, ctol = 1e-3,
                **kwargs):


    def objective(beta):
          return loss_function(y=y, yhat=predict(X=X, beta=beta)) + regularization(X=X,beta=beta)

    def step(arg):
            i, history, params = arg #,params, X,y, history = arg 
            i += 1

            loss, grads = value_and_grad(objective)(params)

            params = params - ((learning_rate * grads) )

            history = history.at[i].set(loss)

            return i, history, params
    
    def cond(arg):
         i ,history,_ = arg 
         return (i < epochs) & (jnp.abs(history[i-1]-history[i]) > ctol)
    
    i=0
    history = jnp.zeros(epochs).at[0].set(jnp.inf)
    params =  jnp.zeros(shape=(X.shape[1]))


    arg = (i, history, params)

    return while_loop(
        cond,
        step,
        arg
    )[2]


def nlm_fit_adam(X, y, predict, loss_function, regularization= lambda X, beta: 0,
            learning_rate=.003, epochs=1000, ctol=1e-8, **kwargs):
  
    def objective(params):
      yhat = predict(X, params)
      return loss_function(y, yhat) + regularization(X=X, beta=params)

    # Define Adam optimizer with learning rate
    opt = adam(learning_rate)

    # Initialize parameters and history
    params = jnp.zeros(shape=(X.shape[1], 1)) 
    history = jnp.zeros(epochs)
    history = history.at[0].set(jnp.inf)



    def update(args):
        params, opt_state, i, history = args
        i = i+1
        loss, grads = value_and_grad(objective)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = apply_updates(params, updates)
        history = history.at[i].set(loss)
        return params, opt_state, i , history


    # Define loop condition based on epoch and loss tolerance
    def cond_fn(args):
      _, _, i, history = args
      return (i < epochs) & (jnp.abs(history[i - 1] - history[i]) > ctol)

    # Use while_loop with custom loop condition and update function
    params, _, _, history = while_loop(
        cond_fn, update, (params, opt.init(params), 0, history),
        )

    return params


from statjax.util import l2, l1
from statjax.metrics import mse

def enet_regularization(l1_penalty, l2_penalty) :

    def calc (X, beta, w = 1.0):

        n = X.shape[0]
        beta_adj = beta * w 

        return (l1_penalty * l1(beta_adj) + l2_penalty * l2(beta_adj))/n

    return jit(calc)

class NLM(LinearModel):
    def __init__(self, predict, loss_function, fit=nlm_fit_adam, regularization = lambda **kwargs:  0, **kwargs):
        regularization = partial(regularization, **kwargs)

        fit = partial(fit, predict = predict, loss_function = loss_function, regularization=regularization, **kwargs)

        super().__init__(fit, predict, cov_filler)

# def NLM(predict, loss_function, fit=nlm_fit_adam, regularization = lambda **kwargs:  0, **kwargs):
#     regularization = partial(regularization, **kwargs)

#     fit = partial(fit, predict = predict, loss_function = loss_function, regularization=regularization, **kwargs)

#     return LinearModel(fit, predict, cov_filler)

def ElasticNet(l1_penalty, l2_penalty, **kwargs):
    return NLM(predict_ols, mse, nlm_fit, regularization=enet_regularization(l1_penalty, l2_penalty, **kwargs))

def LASSO(l1_penalty, **kwargs):
    return ElasticNet(l1_penalty, 0, **kwargs)