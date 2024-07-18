import jax.numpy as jnp 
import jax.random as random
from jax.nn import one_hot as jnn_one_hot
from jax import jit, vmap

import pandas as pd
from formulaic import ModelMatrix, model_matrix

import numpy
from typing import List
from warnings import warn

'''
one_hot: This function takes an array and returns a one-hot encoded matrix.

Almost uniquely, it accepts non-jnp inputs. This is to enable the one-hot encoding of string groups/similar. 

arguments:

arr (iterable): The target to be one-hot encoded.
return_map (bool): If True, the function returns a dictionary mapping the unique values to their respective one-hot indices.

returns:

out (jnp.array): The one-hot encoded matrix.


'''
def one_hot(arr, return_map = False):
    unique_values, indices = numpy.unique(arr, return_inverse=True, axis = 0)

    out = jnn_one_hot(indices, len(unique_values))
    if return_map:
        map = dict(zip(unique_values, range(len(unique_values))))
        return out,map
    else:
        return out
    


'''
add_intercept: This function takes a matrix and adds a column of ones to the left of the matrix.

arguments:

arr (jnp.array): The target matrix.

returns:

 (jnp.array): The matrix with a column of ones added to the left.

'''
def add_intercept(arr):
    arr = validate_array_input(arr)
    return jnp.hstack([jnp.ones((arr.shape[0], 1)), arr])


'''
bootstrap: This function takes a function and a set of arguments, and returns the bootstrap distribution of the statistic.

statistic (callable): The function to be bootstrapped. Any non-array inputs should be partialed in before passing statistic to bootstrap.
                      Must return a scalar or jnp array. 

B (int): The number of bootstrap samples to be taken.

rng (jax.random.PRNGKey): The random key to be used for the bootstrap sampling.

method (str): The method to be used for bootstrapping: "linear" or "vmap".

**method_args: The arguments to be passed to the statistic function. Must have keywords associated with them, and must be arrays of the same shape.
               i.e. X =(jnp.array), Y=(jnp.array), D=(jnp.array).
        

returns:

(jnp.array): The bootstrap distribution of the statistic.


'''

def bootstrap(statistic, B = 1000, rng = random.PRNGKey(0), method = "linear", **method_args):


    a1 = list(method_args.keys())[0]
    n = method_args[a1].shape[0]
    samples = random.choice(rng, n, shape=(B, n), replace=True)

    bootstrap_args = []
    for arg in method_args:
        bootstrap_args.append(jnp.array(method_args[arg][samples]))

    if method == "linear":
        bootstrap_distribution = jnp.array([statistic(*a) for a in (zip(*bootstrap_args))])

    elif method == "vmap":
        bootstrap_distribution = vmap(statistic)(*bootstrap_args)
        
    else:
        raise NotImplementedError("Method must be one of 'linear' or'vmap'")


    return bootstrap_distribution




def ci(bootstrap_distribution, alpha = .05):    
    return jnp.percentile(bootstrap_distribution, jnp.array([alpha/2*100, (1-alpha/2)*100]), axis=0)



'''
residualize: This function takes a control matrix, a target matrix, a model, and a standardize parameter, and returns the residualized target matrix.

control (jnp.array): The control matrix. This matrix is used to predict the target matrix.

target (jnp.array): The target matrix. This matrix is residualized.

model : The model to be used to predict the target matrix. Must have a fit_predict method that takes two matrices and returns a prediction.

standardize (bool): If True, the target matrix is standardized before being returned.

returns:

(jnp.array): The residualized target matrix.

'''

def residualize(control, target,model, standardize = False):
    control = validate_array_input(control)
    target = validate_array_input(target)

    standardize = float(standardize)
    n,k = target.shape

    for idx in range(k):
        y = target[:, idx]
        match = jnp.any(control != y[:, jnp.newaxis], axis=0).astype(int)
        X = control * match
        yhat = model.fit_predict(X,y)
        
        target = target.at[:, idx].set(y - yhat)     

    standardized = (target - jnp.mean(target, axis=0)) / jnp.std(target, axis=0)

    target = target* (1 - standardize) + standardized * standardize


    return target

'''
mundlak: This function takes a target matrix and a nuisance matrix, and returns the Mundlak residuals.
It is equivalent to residualize(one_hot(nuisances), target).

arguments:

target (jnp.array): The target matrix.

nuisances (jnp.array): The nuisance matrix.

returns:

(jnp.array): The Mundlak residuals.

'''

def mundlak(target, nuisances):
    D = one_hot(nuisances)
    # since it's just a group mean, no need to allow for fancier models

    target = validate_array_input(target)
    
    projection = D @ jnp.linalg.pinv(D.T @ D) @ D.T @ target
    return target - projection


'''
IO methods. parse_input is holdover from when models stored dataframes. 

validate_array_input is ligher weight and should be used in the future.



'''

def parse_input(input: jnp.array | List[str], df: pd.DataFrame) -> jnp.array:
    if isinstance(input, jnp.ndarray): # X is a jnp.ndarray
        jnp_output = input

    elif df is not None: # df and non-jnp.ndarray X
        jnp_output = jnp.array(df[input].values)

    elif df is None and not isinstance(input, jnp.ndarray): # no df, non-jnp.ndarray X
        jnp_output = jnp.array(input) # this will throw error if conversion impossible
        warn("arraylike argument is not a jnp.ndarray: converting to jnp.ndarray.")

    else:
        raise ValueError("Could not parse input. Please pass a jnp.ndarray or a dataframe with the appropriate columns.")
        
    assert isinstance(jnp_output, jnp.ndarray)

    if len(jnp_output.shape) == 1:
        jnp_output = jnp_output.reshape(-1,1)

    return jnp_output


def validate_array_input(arr) -> jnp.array:

    if not isinstance(arr, jnp.ndarray): # X is a jnp.ndarray
         warn("arraylike argument is not a jnp.ndarray: converting to jnp.ndarray.")
    jnp_output = jnp.array(arr) # this will throw error if conversion impossible
       
    
    assert isinstance(jnp_output, jnp.ndarray)

    shape = jnp_output.shape
    if len(shape) == 1: 
        jnp_output = jnp_output.reshape(-1,1)
        warn(f"statjax requires 2d arrays as input. recieved {shape}: reshaping to {jnp_output.shape}.")


    assert len(jnp_output.shape) > 1
    
    return jnp_output
@jit
def l1(a):
    return jnp.sum(jnp.abs(a))

@jit
def l2(a):
    return jnp.sum(a ** 2)



'''
process_input: take a unknown-type array and convert it to a ModelMatrix object. 

X: array-like or ModelMatrix or DataFrame: the input array
filler_var_name: str: the name of the filler variable to use if X has no column names (i.e. x: x0, x1, ...) 
spec_base: str: the base of the formulaic string to use for the ModelMatrix object
spec_transform: str: the transformation to apply to each column of X in the formulaic string
'''
import copy
def process_input(X, filler_var_name, spec_base ="-1 + ", spec_transform="", enforced_spec = "") -> ModelMatrix:
        X = copy.deepcopy(X)
        if isinstance(X, ModelMatrix):
            X_mm = X

        else: # not modelmatrix
            if isinstance(X, pd.Series):
                X = pd.DataFrame(X)
            if isinstance(X, pd.DataFrame): 
                X.columns = X.columns.str.replace(' ', '_') # replace spaces with underscores for formulaic
                spec_base += " + ".join([f"{spec_transform}({col})" for col in X.columns])
            else: # assume numpy array/similar
                if len(X.shape) == 1:
                    X = X.reshape(-1,1)
                    cols = [filler_var_name]
                else:
                    cols = [f"{filler_var_name}{i}" for i in range(X.shape[1])]
                    
                cols = [col.replace(" ", "_") for col in cols] 
                X = pd.DataFrame(X, columns=cols)
                spec_base += " + ".join([f"{spec_transform}({col})" for col in X.columns])
                
            if enforced_spec != "":
                spec_base = enforced_spec
            # either case, now have dataframe X and a formulaic-compatible spec
            X_mm = model_matrix(spec_base, X.astype("float64"))
        return X_mm
                 



