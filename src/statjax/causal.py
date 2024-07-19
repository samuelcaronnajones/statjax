
from types import SimpleNamespace
import jax.numpy as jnp
from src.statjax.util import process_input
from functools import partial
from . import OLS
from copy import deepcopy
from jax import vmap, config
from . glm import BernoulliGLM
config.update("jax_enable_x64", True)
import pandas as pd

def check_overlap(D,X):
    D = process_input(D,"d")
    X = process_input(X,"x")

    treated = X.loc[D.values == 1]
    treated_max = treated.max(axis=0)
    treated_min = treated.min(axis=0)

    return ((X >= treated_min) & (X <= treated_max)).all(axis=1)

class CausalEstimator():

    def fit(self):
        pass
    def ate(self):
        pass
    def cef(self):
        pass

def group_mean(D,Y,group, **kwargs):
    member = jnp.array(D==group, dtype=jnp.float32)
    return jnp.sum(member * Y) / jnp.sum(member)


def expermental_ate(D,Y, **kwargs):
    return group_mean(D,Y,1) - group_mean(D,Y,0)



class ExperimentalEstimator(CausalEstimator):
    def __init__(self,):
        self.base = SimpleNamespace()
    def fit(self, D, Y):
        
        self.D = process_input(D, filler_var_name="d")
        self.Y = process_input(Y, filler_var_name="y")

        D_jnp = self.D.values.ravel()
        Y_jnp = self.Y.values.ravel()
        self.base.cef = vmap(partial(group_mean, D_jnp, Y_jnp))
        self.ate = expermental_ate(D_jnp, Y_jnp)
        return self

    



class RegressionEstimator(CausalEstimator):
    def __init__(self, model = OLS()):
        self.outcome_model = model
    def fit(self, D, X, Y):
        self.D = process_input(D, filler_var_name="d")
        self.X = process_input(X, filler_var_name="x")
        self.Y = process_input(Y, filler_var_name="y")

   
        treated = (self.D.values == 1)
        X_0 = self.X.loc[~treated]
        Y_0 = self.Y.loc[~treated]
        X_1 = self.X.loc[treated]
        Y_1 = self.Y.loc[treated]

        self.model0 = deepcopy(self.outcome_model).fit(X_0, Y_0)
        self.model1 = deepcopy(self.outcome_model).fit(X_1, Y_1)

        self.Y0 = self.model0.predict(self.X.copy())
        self.Y1 = self.model1.predict(self.X.copy())

        self.ate = jnp.mean(self.Y1 - self.Y0)
        return self
    


class PropensityScoreEstimator(CausalEstimator):
    def __init__(self, propensity_model=BernoulliGLM(), delta = 0.1):
        self.propensity_model = propensity_model
        self.delta = delta
        self.expectation = [None, None]

    def fit(self,D, X,y, **kwargs):
        self.X = process_input(X, filler_var_name="x")
        self.y = process_input(y, filler_var_name="y")
        self.D = process_input(D, filler_var_name="D")

        self.propensity_model = self.propensity_model.fit(pd.DataFrame(self.X), self.D)
        self.propensities = self.propensity_model.predict(pd.DataFrame(self.X))


        retained = jnp.array((self.propensities > self.delta) & (self.propensities < 1-self.delta), dtype=float)

        if retained.sum() == 0:
            raise ValueError(f"No samples retained. Consider lowering delta: currently {self.delta}")
        yj = jnp.array(self.y.values.ravel())
        Dj = jnp.array(self.D.values.ravel())
        
        self.expectation[0] = jnp.sum((yj * retained * (1-Dj) /(1- self.propensities) ) )/ jnp.sum(retained) # sphaggetti code to do boolean indexing in jax
        self.expectation[1] = jnp.sum( (yj * retained * (Dj) /(self.propensities) ))/ jnp.sum(retained)


        self.ate =   self.expectation[1] - self.expectation[0]
        return self


class DREstimator():
    def __init__(self, propensity_model = BernoulliGLM(), outcome_model = OLS(), delta = 0.1):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.delta = delta

    def fit(self, D, X, Y):
        self.D = process_input(D, filler_var_name="d")
        self.X = process_input(X, filler_var_name="x")
        self.Y = process_input(Y, filler_var_name="y")


        self.propensity_model = self.propensity_model.fit(pd.DataFrame(self.X), self.D)
        self.propensities = self.propensity_model.predict(pd.DataFrame(self.X))

        retained = jnp.array((self.propensities > self.delta) & (self.propensities < 1-self.delta), dtype=float)

        if retained.sum() == 0:
            raise ValueError(f"No samples retained. Consider lowering delta: currently {self.delta}")
        
        # seperate out treated and untreated samples
        treated = (self.D.values == 1)
        X_0 = self.X.loc[~treated]
        Y_0 = self.Y.loc[~treated]
        X_1 = self.X.loc[treated]
        Y_1 = self.Y.loc[treated]

        # fit outcome models
        self.model0 = deepcopy(self.outcome_model).fit(X_0, Y_0,)
        self.model1 = deepcopy(self.outcome_model).fit(X_1, Y_1)

        # predict outcomes
        self.Y0 = self.model0.predict(self.X.copy()).ravel()
        self.Y1 = self.model1.predict(self.X.copy()).ravel()

        Dj = self.D.values.ravel()
        yj = self.Y.values.ravel()
    
        E0 = ((1-Dj)/(1-self.propensities.ravel()))* (yj - self.Y0)  + self.Y0
        E1 = (Dj/(self.propensities.ravel()))* (yj - self.Y1)  + self.Y1

        effects = (E1 - E0 )*retained
        self.ate = (effects).mean() / retained.mean()

        return self
    
