from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
import jax.numpy as jnp 
import pandas as pd

from .util import one_hot, process_input

from jax.scipy.linalg import solve

from jax import config
config.update("jax_enable_x64", True)

from jax import vmap



class LinearModel(): 
    def __init__(self, fit, predict, cov): # these should be pure jax functions
        self.base_methods = {
            "fit": fit,
            "predict": predict,
            "cov": cov
        }

    def fit(self, X, y, *fit_args, add_intercept = True, **fit_kwargs): # fit to data,

        '''
        This is all to deal with variable-type inputs.  
        '''

        if add_intercept == True:
            spec_base = "1 + "
        else:
            spec_base = "-1 + "
        self.X = process_input(X, filler_var_name="x", spec_base = spec_base)
        self.y = process_input(y, filler_var_name="y")

        X_jnp = jnp.array(self.X.values)
        y_jnp = jnp.array(self.y.values).ravel()



        '''
        Now, to actually fit the model. 
        '''

        self.beta = self.base_methods["fit"](X_jnp, y_jnp, *fit_args, **fit_kwargs)


        cov = self.base_methods["cov"](X_jnp, y_jnp, self.beta,*fit_args, **fit_kwargs)
        if not jnp.isnan(cov).any():
            self.cov = cov
            self.se = jnp.sqrt(jnp.diag(self.cov))

        self.resid = y_jnp - self.predict(X)
        
        return self
    
    
    def predict(self, X) -> jnp.array: # predict from fitted model
        X = process_input(X, filler_var_name="x", enforced_spec = self.X.model_spec)
        
        if X.model_spec != self.X.model_spec:
            raise ValueError("Predictor matrix has different features than those used to fit the model.")
        
        X_jnp = jnp.array(X.values.astype(float))
        return self.base_methods["predict"](X_jnp, self.beta)
    



def fit_ols(X, y):
    return  solve(X.T @ X, X.T @ y) #jnp.linalg.pinv(X.T @ X) @ X.T @ y

def predict_ols(X, beta):
    return X @ beta

def cov_ols(X, y, beta): # this is NOT the usual base ols se estimator
    residuals = y - X @ beta
    sigma2 = residuals.T @ residuals / float((X.shape[0] - X.shape[1]))
    return sigma2 * jnp.linalg.pinv(X.T @ X)

'''
Robust covariance matrix calculation.
Following https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf.
Equivalent to "HC0" in Statsmodels - White's initial sandwich estimator.

arguments:

X (array): The predictor matrix. Shape n x k, where n is the number of observations and k is the number of features.
y (array): The target matrix. Shape n x 1.
beta (array): The parameter matrix. Shape k x 1.
w (scalar | array): The weight array. If passed, shape is n x 1. Default float 1.0 leverages the overloading of * 
                    to broadcast the scalar to the correct shape.

returns:

(array): The covariance matrix of the parameter array. Shape k x k.


'''

def cov_ols_robust(X,y,beta):
    yhat = predict_ols(X,beta)
    residuals = y - yhat
    alpha = X @ jnp.linalg.pinv(X.T @  X) 
    h = jnp.diagonal(alpha @ X.T)
    eh= (residuals**2 * (1-h)**-2)
    return alpha.T * eh  @ alpha


'''
Clustered covariance matrix calculation.
Following https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf, with the finite-sample correction presented at (12).
Equivalent to "cluster" in Statsmodels.

arguments: 

X (array): The predictor matrix. Shape n x k, where n is the number of observations and k is the number of features.
y (array): The target matrix. Shape n x 1.
beta (array): The parameter matrix. Shape k x 1.
G (array): One hot encoded matrix of group assignments. Shape n x g, where g is number of groups. 
           Note that the group assignments must be members of the set range(0,g). 

returns:

(array): The covariance matrix of the parameter array. Shape k x k.

'''

def clustered_cov(X,y,beta,G):
    yhat = predict_ols(X,beta)

    n,k = X.shape
    G = one_hot(G)
    g = G.shape[1]
    c =   (n-1 )/(n-k) * (g)/(g-1) # finite-sample correction.

    u = (jnp.sqrt(c) * (y - yhat) ).reshape(-1,1)

    # The einsum is used to separate the data by group, with each group as element on the new axis.
    Xg = jnp.einsum("ij,ik->kij",X,G) 
    ug = jnp.einsum("ij,ik->kij",u,G)
    # print(X.shape)
    
    # The vmap vectorizes the matmul over the groups to calculate the beta_g matricies
    betag = vmap(lambda X,u: X.T @ u @ u.T @ X)(Xg, ug)
    
    # then beta_clu is the sum. From here, standard covariance  matrix calculation.
    beta_clu = betag.sum(axis=0)
    alpha =   jnp.linalg.pinv((X.T) @  X) 
    
    cov = alpha.T @ (beta_clu) @ alpha

    return cov

def fit_ols_clustered(X, y, G):
    return fit_ols(X, y)


ols_cov_map = {
    "non-robust": cov_ols, # LinearModel(fit_ols, predict_ols, cov_ols),
    "robust":cov_ols_robust, # LinearModel(fit_ols, predict_ols, cov_ols_robust),
    "clustered": clustered_cov # LinearModel(fit_ols_clustered, predict_ols, clustered_cov)
}

class OLS(LinearModel):
    def __init__(self, cov_type = "non-robust"):
# def OLS(cov_type = "non-robust"):
        try:
            cov = ols_cov_map[cov_type]
        except KeyError:
            raise ValueError("Only 'non-robust', 'robust', and 'clustered covariance' are currently supported.")
        if cov_type == "clustered":
            fit = fit_ols_clustered
        else:
            fit = fit_ols
        return super().__init__(fit, predict_ols, cov)
    

def fit_ridge(X,y,lam, regularize_intercept):
    lam_matrix = jnp.eye(X.shape[1])* float(lam) # following 181 book
    lam_matrix = lam_matrix.at[0,0].set(lam_matrix[0,0] * (float(regularize_intercept)))
    return solve(X.T @ X + lam_matrix,  X.T @ y)


def cov_filler(X,y,beta):
    return jnp.eye(X.shape[1]) * jnp.nan


class Ridge(LinearModel):
    def __init__(self, lam, regularize_intercept = True):
        cfit = lambda X,y: fit_ridge(X,y,lam, regularize_intercept)
        ccov = cov_filler 
        self.lam = lam
        return super().__init__(cfit, predict_ols, ccov)