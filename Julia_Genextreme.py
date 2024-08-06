# ===============================================================
# CALL JULIA PACKAGE 
# ===============================================================
from julia.api import Julia
from julia import Main
jl = Julia(compiled_modules=False)
# use for load celds code in julia
# %load_ext julia.magic
# Import Julia Packages
try:
    jl.using("Base")
    jl.using("BlackBoxOptim")
    jl.using("Distributions")
    jl.using("DataFrames")
    jl.using("Pandas")
    jl.using("StatsBase")
    jl.using("CSV")
    jl.using("Interpolations")
    jl.using("Dates")
    jl.using("PyCall")
except:
    jl.using("Base")
    jl.using("BlackBoxOptim")
    jl.using("Distributions")
    jl.using("DataFrames")
    jl.using("Pandas")
    jl.using("StatsBase")
    jl.using("CSV")
    jl.using("Interpolations")
    jl.using("Dates")
    jl.using("PyCall")

import pandas as pd
# ===============================================================

class Julia_Genextreme:
    """
    Class for fitting a Generalized Extreme Value (GEV) distribution to a dataset using Julia.
    
    Utilizes optimization via the `Bboptimize` package and the `logpdf` function from the `Distributions` package in Julia
    to find optimal parameters for the GEV distribution. Optimization is performed over location, scale, and shape parameters.

    Attributes:
    ----------
    data_series : pd.Series
        Data series to be fitted to the GEV distribution.
    log_likelihood : float
        Log-likelihood value obtained after fitting the model.

    Methods:
    -------
    fit():
        Performs optimization of the GEV distribution parameters and returns a DataFrame with fitted parameters.
    
    get_aic():
        Calculates and returns the Akaike Information Criterion (AIC) for the fitted model.
    """
    
    def __init__(self, data_series):
        """
        Initializes the class instance with a data series.

        Parameters:
        ----------
        data_series : pd.Series
            Data series to be fitted to the GEV distribution.
        """
        self.data_series = data_series
        self.params = None
        self.log_likelihood = None
    
    def fit(self):
        """
        Performs optimization of the Generalized Extreme Value (GEV) distribution parameters and returns a DataFrame.
        
        Uses the `Bboptimize` package to find optimal values for the location, scale, and shape parameters.
        The optimization function maximizes the negative log-likelihood of the data under the GEV model.

        Returns:
        -------
        pd.DataFrame
            DataFrame with columns for each fitted parameter: 'Shape', 'Location', 'Scale', 
            and with a row index named 'params_df'.
        
        Raises:
        -------
        RuntimeError
            If there is an issue during the Julia evaluation or optimization process.
        """
        # Pass the data series to a Julia array and include it in Julia's environment
        Main.data_max = self.data_series.values

        # Define the objective function for optimization
        Main.eval("""
        function optfun(p, y)
            -sum(map(z -> logpdf(GeneralizedExtremeValue(p[1], abs(p[2]), p[3]), z), y))
        end
        """)

        # Define the search ranges for the optimization and perform the optimization
        Main.eval("""
        res = bboptimize(p -> optfun(p, data_max); SearchRange = [(0.0, 200.0), (0.0, 200.0), (-5.0, 5.0)], Method = :adaptive_de_rand_1_bin, TraceMode=:silent)
        """)
        
        # Extract the optimal parameters and best log-likelihood from the optimization
        loc, scale, shape = Main.eval("best_candidate(res)")
        self.log_likelihood = Main.eval("best_fitness(res)")
        
        # Convert the shape parameter to be compatible with `scipy`
        self.params = [-shape, loc, scale]

        # Create and return a DataFrame with the fitted parameters
        params_df = pd.DataFrame([self.params], columns=['Shape', 'Location', 'Scale'], index=['params_df'])
        return params_df

    def get_aic(self):
        """
        Calculates and returns the Akaike Information Criterion (AIC) for the fitted model.

        Returns:
        -------
        float
            The AIC value calculated for the fitted model.

        Raises:
        -------
        ValueError
            If the `fit()` method has not been called before trying to calculate the AIC.
        """
        if self.params is None or self.log_likelihood is None:
            raise ValueError("The fit() method must be called before calculating the AIC.")
        
        # Number of parameters in the GEV model
        num_params = 3
        # Calculate AIC
        aic = 2 * num_params - 2 * self.log_likelihood
        return aic