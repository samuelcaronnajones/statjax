
from typing import Callable

from flax.linen import Dense, relu, compact, Module, sigmoid
import jax.numpy as jnp
from jax import jit, value_and_grad, random
from jax.lax import while_loop
import optax

import pandas as pd
from formulaic import model_matrix
from formulaic.model_matrix import ModelMatrix
from statjax.metrics import mse
from functools import partial

from . util import process_input

class FlexibleMLP(Module):
    features: tuple  # Tuple of integers representing the number of neurons in each layer
    dropout_rate: float = 0.
    output_activation: Callable = lambda x: x  # Default to identity function

    @compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = Dense(feat)(x)
            
            # Apply ReLU activation to all layers except the last one
            if i < len(self.features) - 1:
                x = relu(x)
    
        # Apply the custom output activation function
        return self.output_activation(x).ravel()

# Example usage
def create_mlp(input_dim, hidden_layers, output_dim, output_activation=lambda x: x):
    layer_sizes = (input_dim,) + hidden_layers + (output_dim,)
    return FlexibleMLP(features=layer_sizes, output_activation=output_activation) 

def nn_fit(X,y, model,loss_function, regularization = lambda X,beta : 0,  epochs= 1000, optimizer = optax.adam(learning_rate = 0.01), ctol = 1e-3, init_key = random.PRNGKey(42)):
    # Define loop condition based on epoch and loss tolerance

    def objective(params):
      yhat = model.apply(params, X)
      return loss_function(y, yhat) + regularization(X=X, beta=params)
    
    step = jit(value_and_grad(objective))

    def update(args):
        params, opt_state, i, history = args
        loss, grads = step(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history = history.at[i].set(loss)
        return params, opt_state, i+1, history

    def cond_fn(args):
      _, _, i, history = args
      return (i < epochs) & (jnp.abs(history[i - 1] - history[i]) > ctol)
    
    # initialize arguments

    params = model.init(init_key, X)
    opt_state = optimizer.init(params)

    history = jnp.zeros(epochs)
    history = history.at[0].set(objective(params))
    args = (params, opt_state, 0, history)

    # run loop

    params, opt_state, i, history = while_loop(cond_fn, update, args)
    return params, history



class NNRegression():
    def __init__(self, output_dim= 1, output_activation=lambda x: x, loss = mse,regularization = lambda X,beta: 0, hidden_layers = (128, 64), optimizer = optax.adam(1e-3), x_transform = "scale" ):
        self.model = partial(create_mlp, hidden_layers = hidden_layers, output_dim = output_dim, output_activation = output_activation)
        self.optimizer = optimizer
        self.loss_fn = loss
        self.params = None 
        self.regularization = regularization
        self.x_transform = x_transform

    def fit(self, X,y, **kwargs):

        '''
        This is all to deal with variable-type inputs.  
        '''
        
        self.X = process_input(X, filler_var_name="x", spec_transform=self.x_transform)
        self.y = process_input(y, filler_var_name="y")
        X_jnp = jnp.array(self.X.values)
        y_jnp = jnp.array(self.y.values).ravel()

        self.model = self.model(X_jnp.shape[1])

        params, history = nn_fit(X_jnp, y_jnp, self.model, self.loss_fn, self.regularization, **kwargs)
        self.params = params
        self.history = history

        self.resid = y_jnp - self.predict(self.X)

        return self

    def predict(self, X):
        X = process_input(X, filler_var_name="x", enforced_spec = self.X.model_spec)
        
        
        if X.model_spec != self.X.model_spec:
            raise ValueError("Predictor matrix has different features than those used to fit the model.")
        
        return self.model.apply(self.params, X)