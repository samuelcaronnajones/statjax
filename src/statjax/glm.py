from oryx.core import inverse, custom_inverse
from jax import hessian, grad, vmap, jit
from functools import partial
import jax.numpy as jnp
from .core import LinearModel
import pandas as pd
from formulaic import ModelMatrix,model_matrix
from jax.scipy.special import erf, erfinv

from jax.nn import sigmoid

from oryx.distributions import Normal, Bernoulli, Poisson, Gamma, InverseGaussian

from jax import config
config.update("jax_enable_x64", True)

ve = 1e-8


def nll_glm(X,y,inverse_link, dist, *params):
    lst_params = list(params)

    mu = inverse_link(X @ params[0])
    lst_params[0] = mu
    ll = dist(*lst_params).log_prob(y.ravel())
    return -jnp.sum(ll)


def glm_J(X,y, link, dist, params):
    param_list = list(params)
    beta = param_list[0]
    eta = X @ beta

    def mu(eta):
        return inverse(link)(eta)
    param_list[0] = mu(eta)

    v = dist(*param_list).variance()
    d_eta= vmap((grad(mu)))(eta) 
    w = d_eta**2 / v

    return X.T * w @ X


def fit_glm_newton_raphson(X,y, link, dist, params, ctol = 1e-3, epochs=100):

    list_params = list(params)
    inv_link = inverse(link)
    def step(index, dL, X,y,*params):
       
        H = hessian(nll_glm,argnums =( 4+index))(X,y, inv_link, dist, *params) 
        return params[index] - jnp.linalg.pinv(H) @ dL(*params)
    
    loss_prev = -1

    for i in range(len(params)):
        dL = lambda *params: grad(nll_glm,argnums =( 4+i))(X,y, inv_link, dist, *params) 

        step_i = jit(partial(step, i, dL))

        for j in range(epochs):
            list_params[i] = step_i(X,y,*list_params)
            loss = nll_glm(X,y, inv_link, dist, *list_params)
            if jnp.abs(loss_prev - loss) < ctol :
                break
            else:
                loss_prev = loss
    

    return tuple(list_params)


def cov_glm(X,y, link, dist, params):
    return jnp.linalg.inv(glm_J(X,y, link, dist, params))

from jax import value_and_grad
from jax.scipy.linalg import solve
from jax.lax import while_loop


def fit_glm_gradient(X,y, link, dist, params, ctol = 1e-3, epochs=100 ):

    list_params = list(params)
    inv_link = inverse(link)

    history = jnp.zeros(epochs)
    history = history.at[0].set(jnp.inf)

    def update_beta_fisher(args):
        beta,  i, history = args
        i += 1
        p = [beta] + list_params[1:]
        loss, dL =  value_and_grad(nll_glm, argnums=(4))(X,y, inv_link, dist, *p)

        J = glm_J(X,y, link, dist, p)
        beta = solve(J, (J @ beta) - dL)


        history = history.at[i].set(loss)
        return beta, i , history
    

    def update_nr(index, args):
        param,  i, history = args
        i += 1

        loss, dL =  value_and_grad(nll_glm, argnums=(4+index))(X,y, inv_link, dist, *list_params[:index], param, *list_params[index+1:])
        H =  hessian(nll_glm,argnums =( 4+index))(X,y, inv_link, dist, *list_params[:index], param, *list_params[index+1:]) #glm_J(X,y, link, dist, p)#


        param = solve(H, H @ param - dL)
        history = history.at[i].set(loss)
        return param, i , history
    
    # Define loop condition based on epoch and loss tolerance
    def cond_fn(args):
      _,  i, history = args
      return (i < epochs) & (jnp.abs(history[i - 1] - history[i]) > ctol)

    
    # beta_init = jnp.linalg.pinv(X.T @ X) @ X.T @ inverse_link(y)

    beta,niter, _=  while_loop(cond_fn, update_beta_fisher, (list_params[0] , 0, history))
    list_params[0] = beta

    for i in range(1, len(list_params)):
    # for i, param in enumerate(list_params):
        update = (partial(update_nr, i))
        list_params[i] = while_loop(cond_fn, update, (list_params[i], 0, history))[0]

    
    return tuple(list_params)

def fit_glm_ls(X,y, link, dist, params, ctol = 1e-3, epochs=100 ):
    
    @jit
    def fisher_w(eta, mu, list_params):  
        
        list_params = list(params)[1:]

        v = dist(mu, *list_params).variance()
        d_eta= vmap((grad(inv_link)))(eta) 
        w = d_eta**2 / v

        return w
    

    list_params = list(params)
    inv_link = inverse(link)

    history = jnp.zeros(epochs)
    history = history.at[0].set(jnp.inf)
    eta_init = inv_link(y)
    w = fisher_w(eta_init, y, list_params)

    beta_init = solve((X.T *w) @ X, (X.T *w ) @ inverse_link(y))
    
    @jit 
    def update_beta_ls(args):
        beta, i, history = args
        i += 1
        eta = X @ beta
        mu  = inv_link(eta)
        z = eta + (y -inv_link(eta)) * vmap(grad(link))(mu)
        w = fisher_w(eta, mu, list_params)

        beta = solve((X.T * w) @ X, (X.T * w) @ z)
        loss = nll_glm(X,y, inv_link, dist, beta, *(list_params[1:]))
        history = history.at[i].set(loss)
        return beta, i, history


    def update_nr(index, args):
        param,  i, history = args
        i += 1

        loss, dL =  value_and_grad(nll_glm, argnums=(4+index))(X,y, inv_link, dist, *list_params[:index], param, *list_params[index+1:])
        H =  hessian(nll_glm,argnums =( 4+index))(X,y, inv_link, dist, *list_params[:index], param, *list_params[index+1:]) #glm_J(X,y, link, dist, p)#

        param = solve(H, H @ param - dL)
        history = history.at[i].set(loss)
        return param, i , history
    


    # Define loop condition based on epoch and loss tolerance
    def cond_fn(args):
      _,  i, history = args
      return (i < epochs) & (jnp.abs(history[i - 1] - history[i]) > ctol)

    

    beta,niter, _=  while_loop(cond_fn, update_beta_ls, (beta_init , 0, history))
    list_params[0] = beta

    for i in range(1, len(list_params)):
    # for i, param in enumerate(list_params):
        update = (partial(update_nr, i))
        list_params[i] = while_loop(cond_fn, update, (list_params[i], 0, history))[0]
        

    
    return tuple(list_params)

class GLM(LinearModel): 
    def __init__(self, link, dist, params_init, fit = fit_glm_ls, **fit_kwargs):

        def predict(X,beta):
            return inverse(link)(X @ beta)
        
        if not isinstance(params_init, tuple):
            params_init = (params_init,)
            
        self.base = {
            "link": link,
            "dist": dist,
            "cov": cov_glm,
            "params_init": params_init,
            "fit": partial(fit, link=link, dist=dist, params=params_init,**fit_kwargs),
            "cov": partial(cov_glm, link = link, dist = dist),
            "predict": predict
        }

    def fit(self, X, y, add_intercept = True): # fit to data,

        '''
        This is all to deal with variable-type inputs.  
        '''

        if isinstance(X, ModelMatrix):
            if add_intercept:
                raise ValueError("Cannot add intercept to existing ModelMatrix. Please add intercept before creating ModelMatrix.")
            X_jnp = X.values
            self.X = X

        elif not isinstance(X, ModelMatrix):
            if isinstance(X, pd.DataFrame): 
                spec_base =   " + ".join(X.columns)

            else:
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
                cols = [f"x{i}" for i in range(1, X.shape[1] + 1)]
                X = pd.DataFrame(X, columns=cols)
                spec_base = " + ".join(cols)

            if not add_intercept:
                spec_base = "-1 + " + spec_base

            self.X = model_matrix(spec_base, X.astype("float64"))
            X_jnp = jnp.array(self.X.values)

        if isinstance(y, ModelMatrix):
            self.y = y

        elif not isinstance(y, ModelMatrix):

            if isinstance(y, pd.DataFrame): 
                assert len(y.columns) == 1
                y_code = y.columns[0]
            elif isinstance(y, pd.Series):
                y_code = y.name
                y = pd.DataFrame(y)
            else:
                y_code = "y"
                y = pd.DataFrame(y, columns=[y_code])
                
            
            self.y = model_matrix(y_code + "-1", y)
        
        y_jnp = jnp.array(self.y.values.astype(float)).ravel()




        '''
        Now, to actually fit the model. 
        '''

        self.params = self.base["fit"](X_jnp, y_jnp,)

        
        self.beta = self.params[0]
        self.cov = self.base["cov"](X=X_jnp,y=y_jnp, params=self.params)
        self.se = jnp.sqrt(jnp.diag(self.cov))
        self.resid = y_jnp - self.predict(X)

        self.nll = nll_glm(X_jnp,y_jnp, self.base["link"], self.base["dist"], *self.params)
        self.AIC = 2 * X_jnp.shape[1] + 2 * self.nll
        self.BIC =jnp.log(X_jnp.shape[0])  * X_jnp.shape[1] + 2 * self.nll
        
        return self
    
    
    def predict(self, X) -> jnp.array: # predict from fitted model
        if not isinstance(X, ModelMatrix):
            if isinstance(X, pd.DataFrame): 
                X = model_matrix(self.X.model_spec, X)
            else:
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
                cols = [f"x{i}" for i in range(1, X.shape[1] + 1)]
                X = pd.DataFrame(X, columns=cols)
                X = model_matrix(self.X.model_spec, X)
        
        if X.model_spec != self.X.model_spec:
            raise ValueError("Predictor matrix has different features than those used to fit the model.")
        
        X_jnp = jnp.array(X.values.astype(float))
        return self.base["predict"](X_jnp, self.beta)
    


def identity_link (x):
    return x

def log_link(mu):
    return jnp.log(mu)

@custom_inverse
def logit_link(mu):
    return mu / (1-mu)
logit_link.def_inverse_unary(lambda mu: sigmoid(mu),f_ildj = lambda mu: 0)



def probit_link_inverse(mu):
    return Normal(0., 1.).cdf(mu)
@custom_inverse
def probit_link(mu):
    mu = jnp.clip(mu, 1e-10, 1 - 1e-10)
    return jnp.sqrt(2) * erfinv(2*mu - 1)
probit_link.def_inverse_unary(probit_link_inverse,f_ildj = lambda mu: 0)



# Register the custom inverse function


def inverse_link (mu):
    return  1/mu

def fit_normal_glm(X,y, link, dist, params, ctol = 1e-3, epochs=100):
    if params == (-1,-1):
        params = (jnp.zeros((X.shape[1])), jnp.array([1.0]))
    return fit_glm_ls(X,y, link, dist, params, ctol, epochs)


class NormalGLM(GLM):
    def __init__(self, link = identity_link, **kwargs):
        super().__init__(link, Normal,  (-1,-1), fit = fit_normal_glm, **kwargs)

def fit_logit_glm(X,y, link, dist, params, ctol = 1.0, epochs=100):

    if params == (-1,):
        params = (jnp.zeros((X.shape[1])), )
    return fit_glm_gradient(X,y, link, dist, params, ctol, epochs)


class BernoulliGLM(GLM):
    def __init__(self, link  = logit_link, **kwargs):
        return super().__init__(link, Bernoulli, -1, fit = fit_logit_glm, **kwargs)

def fit_poisson_glm(X,y, link, dist, params, epochs=100, **kwargs):

    if params == (-1.0,):
        if link == log_link:
            params = (jnp.zeros((X.shape[1]))+ .001 )
        else:
            params = (jnp.ones((X.shape[1])) * ve)

    return fit_glm_gradient(X,y, link, dist, (params,), ctol = 1e-5, epochs=epochs)


class PoissonGLM(GLM):
    def __init__(self, link = log_link, **kwargs):
        super().__init__(link, Poisson, (-1.0,), fit = fit_poisson_glm, **kwargs)


def  gamma_nef(mu, concentration):
    rate = concentration / mu

    return Gamma(concentration, rate)


def fit_gamma_glm(X,y, link, dist, params):

    if params == (-1.0, -1.0):
        params = (jnp.zeros((X.shape[1])) + ve, jnp.ones(1))

    return fit_glm_ls(X,y, link, dist, params, ctol = 1e-3, epochs=100)

class GammaGLM(GLM):
    def __init__(self, link = inverse_link, **kwargs):
        return super().__init__(link, gamma_nef, (-1.0, -1.0), fit = fit_gamma_glm, **kwargs)



def inverse_squared_link(mu):
    return  mu **(-2) 

def fit_inverse_gaussian_glm(X,y, link, dist, params):

    if params == (-1.0, -1.0):
        params = (jnp.zeros((X.shape[1])) + ve, jnp.ones(1) )

    return fit_glm_gradient(X,y, link, dist, params, ctol = 1e-3, epochs=100)

class InverseNormalGLM(GLM):
    def __init__(self, link = inverse_squared_link, **kwargs):
        return super().__init__(link, InverseGaussian, (-1.0, -1.0), fit = fit_inverse_gaussian_glm, **kwargs)





