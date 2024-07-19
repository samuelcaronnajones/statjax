# statjax

This library is my attempt to compile the data science tools that I use day-to-day in a single place. The main convenience feature over `sklearn` or `statsmodels` is a port of the Python `stargazer` package that can produce latex tables displaying any of the linear models in the package side-by-side. It duplicates part of the `statsmodels` GLM functionality, and provides a general GLM class that the user can initialize with an arbitrary link function and `oryx` distribution. It also provides a handful of ATE estimators. 

All of the models can take array, dataframe/series, or `Formulaic` modelmatricies as arguments. 

The backend of the package is written in `jax`. Overhead is higher, but the package will outperform `statsmodels` and `sklearn` in large-sample or high-dimensional cases. 