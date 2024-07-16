import jax.numpy as jnp 
from jax import jit


'''
mse: compute mean squared error.

Args:
y (array): actual values.
yhat (array): predicted values.

Returns:
float: mean squared error
'''

@jit
def mse(y, yhat):
    return jnp.mean((y-yhat) ** 2)

'''
mae: compute mean absolute error.

Args:
y (array): actual values.
yhat (array): predicted values.

Returns:
float: mean squared error
'''

@jit
def mae(y,yhat):
    return jnp.mean(jnp.abs(y-yhat))      

'''
r2: compute the R-squared value.

Args:
y (array): actual values.
yhat (array): predicted values.

Returns:
float: R-squared value
'''
@jit
def r2(y,yhat):
    mss = jnp.sum((y-jnp.mean(y))**2)
    rss = jnp.sum((y-yhat)**2)
    return 1 - (rss/mss)



'''
adj_r2: compute the adjusted R-squared value.
Args:
y (array): actual values.
yhat (array): predicted values.
X (array): feature dataset. Used to generate number of predictors.

Returns:
float: adjusted R-squared value
'''


@jit
def adj_r2(y,yhat, X):
    n,p = X.shape
    return 1 - (1 - r2(y,yhat)) * (n-1)/(n-p-1)

'''
f_test: Compute the F-statistic for a set of predictions.

Args:
y (array): Actual values.
yhat (array): Predicted values from the regression model.
num_predictors (int): Number of predictors used in the model.
X (array): feature dataset. Used to generate number of predictors.
Returns:
float: F-statistic
'''


@jit
def f_test(y, yhat, X):
    k = X.shape[1]
    # Calculate the residual sum of squares
    sse = jnp.sum((y - yhat) ** 2)
    
    # Calculate the total sum of squares
    sst = jnp.sum((y - jnp.mean(y)) ** 2)
    
    # Calculate the regression sum of squares
    ssr = sst - sse
    
    # Degrees of freedom
    n = X.shape[0]  # Number of observations
    df_regression = k
    df_residual = n - k - 1
    
    # Mean square regression and residual
    msr = ssr / df_regression
    mse = sse / df_residual
    
    # F-statistic
    f_statistic = msr / mse
    
    return f_statistic


"""
Below are metrics associated with binary classification/propensity modeling tasks.


"""





ve =10e-8


@jit
def log_cross_entropy(y,yhat):
        return  -jnp.mean(y * jnp.log(yhat + ve) + (1 - y) * jnp.log(1 - yhat + ve))



def predict_label(yhat, threshold = .5):
    return yhat > threshold

def classification_accuracy(Y, yhat, threshold = .5):

    return jnp.mean(predict_label(yhat, threshold) == Y)

def brier_score(Y, yhat):
    return jnp.mean((yhat - Y) ** 2)

def precision(Y, yhat):
    return jnp.sum(Y * predict_label(yhat)) / jnp.sum(predict_label(yhat))

def recall(Y, yhat):
    return jnp.sum(Y * predict_label(yhat)) / jnp.sum(Y)

def f1_score(Y, yhat):
    prec = precision(Y, yhat)
    rec = recall(Y, yhat)
    return 2 * (prec * rec) / (prec + rec)
