# statjax

This library is my attempt to compile the data science tools that I use day-to-day in a single place while maintaining my coding ability and checking my understanding of various models by implementing them.

 It consists of a handful of causal estimators and a flexible linear model framework. The main convenience feature over Scikit-Learn or Statsmodels is a port of the Python Stargazer package that can produce latex tables displaying any of the linear models in the package side-by-side. The backend of the package is written in Jax and Oryx. Overhead is higher, but the package can outperform other data science libraries in large-sample or high-dimensional cases. 

The probabilistic GLM framework in the package is designed to be very modular, capable of defining GLMS in terms of a passed link function and error distribution. There are two GLM extensions that are potentially novel to Python. The first is elastic net regularization based on Glmnet in R, and the second is the ability to easily generate Bayesian linear models defined by a link function, error distribution, and prior distribution in a more standard API than that offered by PyMC3. The novelty comes from the flexibility of Oryx: rather than restricting the models to a pre-determined family of distributions, any Oryx distribution, including user-defined, can be passed for any of the distribution arguments. Similarly, any user-defined link function can be used to initialize the model, and this flexibility doesn't compromise the simplicity of the API. 

See the demo.ipynb for a full demonstration of the packages functionality.

While the models don't directly support R-style formulas as arguments, all are natively compatible with Formulaic model matrices. Initializing the design matrix using Formulaic then passing those matrices into a Statjax model duplicates the R functionality of passing a formula as model argument at the cost of a single additional line of code. The model will then automatically apply the formula of the design matrix used to fit the model in predict/score/similar functions on dataframe inputs and prevent the user from passing model matrices with different formulas to minimize headaches when iterating on formulas. 



