import jax.numpy as jnp
from jax.scipy.special import betainc


"""
These are util functions for the stargazer since they aren't in jax.scipy yet.
Hopefully, these will be replaced in the future. 
"""


def f_distribution_cdf(x, d1, d2):
    """
    Compute the CDF of the F-distribution using the regularized incomplete beta function.

    Parameters:
    x (jnp.ndarray): The values at which to compute the CDF.
    d1 (int): The degrees of freedom for the numerator.
    d2 (int): The degrees of freedom for the denominator.

    Returns:
    jnp.ndarray: The CDF values for the F-distribution at each value of x.
    """
    # Compute the variable for the beta function
    beta_variable = d1 * x / (d1 * x + d2)
    
    # Calculate the CDF using the regularized incomplete beta function
    cdf_values = betainc(d1 / 2.0, d2 / 2.0, beta_variable)
    
    return cdf_values


def t_distribution_cdf(t, nu):

    
    """
    Correctly compute the CDF of the t-distribution using the regularized incomplete beta function,
    aligning it with scipy.stats.t's implementation.

    Parameters:
    t (float or np.ndarray): The values at which to compute the CDF.
    nu (int): The degrees of freedom of the t-distribution.

    Returns:
    float or np.ndarray: The CDF values for the t-distribution at each value of t.
    """
    # Compute the transformation variable for the beta function
    x = t**2 / (nu + t**2)

    # Compute the CDF using the regularized incomplete beta function
    # The CDF for t >= 0 is computed by taking 1 minus half of the tail probability
    cdf_values = jnp.where(t >= 0,
                          0.5 + 0.5 * betainc(0.5, nu / 2.0, x),
                          0.5 - 0.5 * betainc(0.5, nu / 2.0, x))
    
    return cdf_values

