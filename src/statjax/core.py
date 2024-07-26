from types import SimpleNamespace
import jax.numpy as jnp 
import pandas as pd

from .util import one_hot, process_input
from .metrics import r2

from jax.scipy.linalg import solve

from jax import config
config.update("jax_enable_x64", True)

from jax import vmap
from . tables import RegressionTable



'''
General framework for linear models. 

Callable fit: (LinearModel, jnp.ndarray X, jnp.ndarry y) -> dict[str:value]

    Takes a linearmodel, X, and y, and returns the fitted parameters of the model. For example, in the case of OLS, it would return a dictionary with the key "beta" and the value being the fitted beta, while in the Normal Linear Model, it would return a dictionary with the keys "beta" and "dist_params" and the values being the fitted beta and the fitted scale parameter respectively. 

Callable predict: (LinearModel, jnp.ndarray X) -> jnp.ndarray

    Takes a linearmodel and X, and returns the E[y|X] implied by the model.

Callable score: (LinearModel, jnp.ndarray X, jnp.ndarray y) -> float

    Takes a linearmodel, X, and y, and returns the score of the model. Ensures compatibility with Sklearn CV and other tools in that framework. 
    
dict[str:Callable] saved_stats: 

    A dictionary of statistics that the model should save after the fitting process. Each stat should be of the form Callable(LinearModel) -> value. For example, in the case of OLS, it includes the covariance matrix of the residuals, standard errors of beta from that covariance matrix, and the R^2 of the model. 

    Some standards for RegressionTable compatibility: se should elementwise Z-stat of the beta, ci should be of the shape (2, k) for k-vector beta corresponding to the 95% confidence interval of the beta.

'''
class LinearModel():
    def __init__(self, fit, predict, score, saved_stats = {}, **kwargs):
        self.base = SimpleNamespace(fit=fit, predict=predict, score=score)
        self.saved_stats = saved_stats
        self.__dict__.update(kwargs)

    def fit(self, X, y, *fit_args, add_intercept = True, **fit_kwargs):
        X_spec_base = "-1 + " if not add_intercept else ""
        self.X = process_input(X,filler_var_name="x", spec_base=X_spec_base)
        self.y = process_input(y,filler_var_name="y")
   

        fit_result = self.base.fit(self, self.X.values, self.y.values.ravel(), *fit_args, **fit_kwargs)
        for k in fit_result:
            self.__setattr__(k, fit_result[k])
            
        self.resid = self.y.values.ravel() - self.predict(self.X)

        for s in self.saved_stats:
            self.__setattr__(s, self.saved_stats[s](self))

        return self
    
    def predict(self, X):
        X = process_input(X, filler_var_name="x", enforced_spec=self.X.model_spec)
        return self.base.predict(self, X.values)
    
    def score(self, X, y):
        X = process_input(X,filler_var_name="x", enforced_spec=self.X.model_spec)
        y = process_input(y,filler_var_name="y")
        return self.base.score(self, X.values, y.values.ravel())
    
    def summary(self):
        return RegressionTable(self)
    def get_params(self, deep= False):
        return self.__dict__
    def set_params(self, **params):
        self.__dict__.update(params)
        return self

def ols_predict(model, X):
    return X @ model.beta

def ols_fit(obj, X,y): 
    return {"beta": solve(X.T @ X, X.T @ y)}

def ols_score(obj, X, y):
    yhat = obj.base.predict(obj, X)
    return r2(y,yhat).item()

def ols_cov(obj):
    resid = obj.resid
    X = obj.X.values
    return jnp.linalg.inv(X.T @ X) *jnp.dot(resid, resid) / (X.shape[0] - X.shape[1])

def se_from_cov(obj):
    cov = obj.cov
    return jnp.sqrt(jnp.diag(cov))

def ci_from_z_se(obj):
    beta, se = obj.beta, obj.se
    return jnp.array([beta - 1.96 * se, beta + 1.96 * se])

def fit_r2(obj):
    return r2(obj.y.values.ravel(), obj.predict(obj.X)).item()

def calculate_adjusted_r2(r2_score, n, k):
    adjusted_r2 = 1 - ((1 - r2_score) * (n - 1)) / (n - k - 1)
    return adjusted_r2

def fit_r2_adj(obj):
    r2_score =  r2(obj.y.values.ravel(), obj.predict(obj.X)).item()
    n, k = obj.X.shape
    return calculate_adjusted_r2(r2_score, n, k)



# def fit_ols(X, y):
#     return  solve(X.T @ X, X.T @ y) #jnp.linalg.pinv(X.T @ X) @ X.T @ y

# def predict_ols(X, beta):
#     return X @ beta

def cov_ols(model): # this is NOT the usual base ols se estimator
    X, y, beta = model.X.values, model.y.values.ravel(), model.beta
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

def cov_ols_robust(model):
    X,  residuals = model.X.values, model.resid
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

def clustered_cov(model):
    X, residuals, G = model.X.values, model.resid, model.groups

    n,k = X.shape
    G = one_hot(G)
    g = G.shape[1]
    c =   (n-1 )/(n-k) * (g)/(g-1) # finite-sample correction.

    u = (jnp.sqrt(c) * (residuals) ).reshape(-1,1)

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

def fit_ols_clustered(model, X, y, G):
    model.groups = G
    return ols_fit(model, X, y)


ols_cov_map = {
    "non-robust": cov_ols, # LinearModel(fit_ols, predict_ols, cov_ols),
    "robust":cov_ols_robust, # LinearModel(fit_ols, predict_ols, cov_ols_robust),
    "clustered": clustered_cov # LinearModel(fit_ols_clustered, predict_ols, clustered_cov)
}


class OLS(LinearModel):
    def __init__(self, covariance_type = "non-robust", **kwargs):
        cov = ols_cov_map[covariance_type]
        fit = fit_ols_clustered if covariance_type == "clustered" else ols_fit
        super().__init__(fit, ols_predict, ols_score,
                          saved_stats = {"cov": cov,
                                          "se": se_from_cov,
                                          "ci": ci_from_z_se,
                                          "r2": fit_r2,
                                          "r2_adj": fit_r2_adj},**kwargs)


def ridge_fit(X,y,lam):
    _, k = X.shape
    beta = solve(X.T @ X + lam * jnp.eye(k), X.T @ y)
    return {"beta": beta}

class Ridge(LinearModel):
    def __init__(self, lam,**kwargs):
        self.lam = lam
        def fit(model, X,y):
            return ridge_fit(X,y,model.lam)
        super().__init__(fit, ols_predict, ols_score,
                          saved_stats = {"r2": fit_r2,
                                          "r2_adj": fit_r2_adj},**kwargs)
