# import data analysis libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme(style='ticks')

# import stats libraries
from fitter import Fitter
from scipy.stats import genextreme
from scipy.optimize import least_squares
from statsmodels.distributions.empirical_distribution import ECDF
from Julia_Genextreme import Julia_Genextreme
from scipy.optimize import curve_fit
import scipy.stats as stats
from math import ceil
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
# Ignore warnings
warnings.filterwarnings("ignore")

class IDFAnalysis:
    """
    A class for performing Intensity-Duration-Frequency (IDF) analysis on rainfall data.
    
    This class encapsulates methods for calculating annual maximum intensities,
    fitting statistical models, generating IDF curves, and plotting results.
    """

    def __init__(self, historic_hourly, Durations, Return_periods, distribution='genextreme', model='scipy_stats', method='curve_fit', IDF_type='IDF_typeI', scale = 'normal'):
        """
        Initialize the IDFAnalysis class.
        
        Args:
            historic_hourly (pd.DataFrame): Historical hourly rainfall data.
            Durations (np.array): Array of durations to analyze (in hours)
            Return_periods (np.array): Array of return periods to calculate.
            distribution (str): Statistical distribution to use for fitting. Default is 'genextreme'.
            model (str): Model engine to use for fitting. Options are 'scipy_stats' or 'Julia_stats'. Default is 'scipy_stats'.
        """
        self.historic_hourly = historic_hourly
        self.Durations = Durations
        self.intensity_annual_max_dict = {}
        self.modelos_fit = {}
        self.Gages_idf = {}
        self.Return_period = Return_periods
        self.Non_Exceedance_Probability = 1 - 1 / self.Return_period
        self.distribution = distribution
        self.model = model
        self.method = method
        self.IDF_type = IDF_type
        self.scale = scale
        
        # Perform initial calculations
        self._calculate_intensity_annual_max()
        self._fit_models()
        self._calculate_idf()
        
    def _calculate_intensity_annual_max(self):
        """
        Calculate annual maximum intensities for each duration and station.
        """
        for station in self.historic_hourly.columns:
            int_h = pd.DataFrame(index=self.historic_hourly.index, columns=self.Durations)
            for duration in self.Durations:
                # Calculate rolling sum for each duration
                pcp_ = self.historic_hourly[station].rolling(duration, center=True).sum()
                # Convert to intensity (mm/hour)
                int_h[duration] = pcp_ / duration
            # Get annual maximum for each duration
            annualmax = int_h.groupby(by=int_h.index.year).max()
            self.intensity_annual_max_dict[station] = annualmax

    def _fit_models(self):
        """
        Fit statistical models to annual maximum intensities.
        """
        self.modelos_fit = self.fit_data()

    def fit_data(self):
        """
        Fit statistical models to the data.
        
        Returns:
            dict: Fitted models for each station and duration.
        """
        modelos_fit = {}
        
        for gage in self.historic_hourly.columns:
            modelos_gage = {}
            
            for duracion in self.Durations:
                data = self.intensity_annual_max_dict[gage][duracion]
                
                if self.model == 'scipy_stats':
                    f = Fitter(data, distributions=[self.distribution])
                    f.fit()
                    results = f.summary(plot=False).sort_values('aic')
                    distr = getattr(stats, results.index[0])
                    params = f.fitted_param[results.index[0]]
                    
                    # Create and save the fitted model
                    modelos_gage[duracion] = distr(*params)
                
                elif self.model == 'Julia_stats':
                    # Assuming Julia_Genextreme is a function that fits the parameters
                    Model_gev_fit_julia = Julia_Genextreme(data)
                    params = Model_gev_fit_julia.fit().values[0]

                    # Create a scipy model with parameters calculated by Julia
                    modelos_gage[duracion] = stats.genextreme(*params)
            
            modelos_fit[gage] = modelos_gage
        
        return modelos_fit

    def _calculate_idf(self):
        """
        Calculate Intensity-Duration-Frequency (IDF) values for each station.
        """
        for station in self.historic_hourly.columns:
            IDF = pd.DataFrame(index=self.Return_period, columns=self.Durations)
            IDF.index.name = 'Tr'
            for duration in self.Durations:
                model = self.modelos_fit[station][duration]
                # Calculate intensity for each return period
                IDF.loc[:, duration] = model.ppf(self.Non_Exceedance_Probability)
            # Sort intensities in descending order for each return period
            df_sorted = IDF.apply(lambda row: sorted(row, reverse=True), axis=1, result_type='expand')
            df_sorted.columns = IDF.columns
            self.Gages_idf[station] = df_sorted.copy()
    
    def get_idf_table(self, station=None):
        """
        Get the IDF table for a specific station or all stations.
        
        Args:
            station (str, optional): The station name to get the IDF table for.
                                     If None, returns IDF tables for all stations.
        
        Returns:
            pd.DataFrame or dict of pd.DataFrames: The IDF table(s) for the specified station(s).
        """
        if station is not None:
            if station in self.Gages_idf:
                return self.Gages_idf[station]
            else:
                raise ValueError(f"Station '{station}' not found in the analysis.")
        else:
            return self.Gages_idf

    def plot_cdf_models(self, station):
        """
        Generate Cumulative Distribution Function (CDF) plots for a specific station.
        
        Args:
            station (str): The station to plot.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        n_plots = len(self.Durations)
        n_cols = min(3, n_plots)
        n_rows = ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows), dpi = 90)
        fig.suptitle(f'CDF Plot - Station {station}', fontsize=16)
        
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for ax, duration in zip(axes, self.Durations):
            data = self.intensity_annual_max_dict[station][duration]
            model = self.modelos_fit[station][duration]
            ecdf = ECDF(data)
            
            I = np.linspace(start=min(data)*0.8, stop=max(data)*1.2, num=1000)
            ax.plot(I, model.cdf(I), label='Fitted model', color='#1f77b4', linewidth=2)
            ax.scatter(data, ecdf(data), s=30, label='Observed data', color='#ff7f0e', alpha=0.7)
            
            ax.set_xlabel('Intensity (mm/h)', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'Duration: {duration}h', fontsize=12)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Hide empty subplots
        for ax in axes[n_plots:]:
            ax.set_visible(False)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_qq_models(self, station):
        """
        Generate Quantile-Quantile (Q-Q) plots for a specific station.
        
        Args:
            station (str): The station to plot.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        n_plots = len(self.Durations)
        n_cols = min(3, n_plots)
        n_rows = ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'Q-Q Plot - Station {station}', fontsize=16)
        
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for ax, duration in zip(axes, self.Durations):
            data = self.intensity_annual_max_dict[station][duration]
            model = self.modelos_fit[station][duration]
            
            theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, len(data)))
            empirical_quantiles = np.sort(data)
            
            ax.scatter(theoretical_quantiles, empirical_quantiles, s=30, color='#2ca02c', alpha=0.7)
            
            # Add reference line y=x
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
            
            ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax.set_ylabel('Empirical Quantiles', fontsize=10)
            ax.set_title(f'Duration: {duration}h', fontsize=12)
            ax.tick_params(labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Hide empty subplots
        for ax in axes[n_plots:]:
            ax.set_visible(False)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    @staticmethod
    def IDF_typeI(x, b, c, d, e):
        """IDF equation type I."""
        a = d * x[0] + e
        I = a / (x[1] + c)**b
        return I

    @staticmethod
    def IDF_typeII(x, b, c, d, e):
        """IDF equation type II."""
        a = d * x[0] + e
        I = a / (x[1]**b + c)
        return I

    @staticmethod
    def IDF_typeIII(x, b, c, d, e):
        """IDF equation type III."""
        a = d * x[0]**e 
        I = a / (x[1] + c)**b
        return I

    @staticmethod
    def IDF_typeIV(x, b, c, d, e):
        """IDF equation type IV."""
        a = d * x[0]**e
        I = a / (x[1]**b + c)
        return I

    @staticmethod
    def IDF_typeV(x, b, c, d, e):
        """IDF equation type V."""
        a = d * x[0]**e
        I = a / (x[1])**b
        return I
    
    @staticmethod
    def IDF_typeV_pot_reg(T, D, d, e, b):
        """
        Calculate the intensity of rainfall based on the given formula.

        Parameters:
        k (float): Regression coefficient.
        m (float): Exponent for return period.
        n (float): Exponent for duration.
        T (float): Return period.
        t (float): Duration in minutes.

        Returns:
        float: Calculated rainfall intensity.
        """
        return (d * (T ** e)) / (D ** b)

    def fit_multi_IDF_curves(self, x, params, func):
        """
        Fit multiple IDF curves simultaneously.
        
        Args:
            x (pd.DataFrame): DataFrame of observed IDF values.
            params (list): List of initial parameters.
            func (callable): Function of the IDF equation for least squares.
            
        Returns:
            np.array: Squared residuals between fitted and observed values.
        """
        y = pd.DataFrame().reindex_like(x)
        for idx, T in enumerate(x.index):
            # Apply IDF_type to each row
            for D in x.columns:
                y.loc[T, D] = func([T, D], params[0], params[idx+1], params[idx+8], params[idx+15])
        return (y-x).values.reshape(-1)**2
    
    def residuo(self, params, IDF, func):
        """
        Calculate residuals for optimization.
        
        Args:
            params (list): List of initial parameters.
            IDF (pd.DataFrame): DataFrame of observed IDF values.
            func (callable): Function of the IDF equation for least squares.
            
        Returns:
            np.array: Squared residuals between fitted and observed values.
        """
        if self.scale == 'log':
            # Apply the logarithmic transformation to the observed IDF values
            log_IDF = np.log(IDF)
            return self.fit_multi_IDF_curves(log_IDF, params, func)
        elif self.scale == 'normal':
            # Work directly with the original values if the scale is normal
            return self.fit_multi_IDF_curves(IDF, params, func)
        else:
            raise ValueError("The 'scale' parameter must be 'normal' or 'log'")
    
    def construct_multi_IDFs(self, Ts, Ds, params, func):
        """
        Construct multiple synthetic IDF curves.
        
        Args:
            Ts (list): List of return periods.
            Ds (list): List of durations.
            params (list): List of optimized parameters.
            func (callable): Function of the IDF equation for least squares.
            
        Returns:
            pd.DataFrame: DataFrame of synthetic IDF curves.
        """
        res = pd.DataFrame(index=Ts, columns=Ds)
        for idx, T in enumerate(res.index):
            for D in res.columns:
                res.loc[T, D] = func([T, D], params[0], params[idx+1], params[idx+8], params[idx+15])
        
        # Convert all values to numeric
        res_numeric = res.apply(pd.to_numeric, errors='coerce')
        
        # Handle based on the specified scale
        if self.scale == 'log':
            # Apply np.exp() to revert to the original scale from logarithmic
            res_numeric = np.exp(res_numeric)
        elif self.scale == 'normal':
            # No additional transformation needed if the scale is normal
            pass
        else:
            raise ValueError("The 'scale' parameter must be 'normal' or 'log'")
        
        return res_numeric

    def IDF_fit(self, station, IDF_type=None, method=None, plot=True):
        """
        Fit the IDF curve for a specific station using the specified method.
        
        Args:
            station (str): Station to analyze.
            IDF_type (str): Type of IDF equation to use.
            method (str): Fitting method to use ('curve_fit' or 'least_squares').
            plot (bool): Whether to generate a plot of the results.

        Returns:
            pd.DataFrame: Fitted IDF curve data.
            matplotlib.figure.Figure: Generated figure (if plot=True).
        """
        if method is None:
            method = self.method  # Use class default method if not provided
        if IDF_type is None:
            IDF_type = self.IDF_type  # Use class default IDF_type if not provided
        
        station_idf = self.Gages_idf[station]
        
        # Point duration
        (RR, DD) = np.meshgrid(self.Return_period, self.Durations)
        Duracion_puntual = np.vstack([RR.reshape(-1), DD.reshape(-1)])
        
        # Intensity
        Intensidad_puntual = station_idf.melt()['value'].values
        # From 15 minutes to 24 hours
        Duracion_lluvia = np.linspace(0.0833333, self.Durations.max(), 500)
        # Include in fit serie all durations that are not already in the IDF curve
        Duracion_lluvia = np.unique(np.concatenate((Duracion_lluvia, self.Durations)))
        
        if method == 'curve_fit':
            # Fit curve using curve_fit
            params, _ = curve_fit(getattr(self, IDF_type), Duracion_puntual, Intensidad_puntual, maxfev=10000)
            
            IDF_curve_fit = pd.DataFrame(index = Duracion_lluvia)
            for tr in self.Return_period:
                Intensidad_lluvia = [getattr(self, IDF_type)((tr, duracion), *params) for duracion in Duracion_lluvia]
                IDF_curve_fit[tr] = Intensidad_lluvia
            
        elif method == 'least_squares':
            # Fit curve using least_squares
            num_periods = len(self.Return_period)
            num_parameters = 4 # number parameters to fit in idf equations
            ajuste = least_squares(self.residuo, x0=1.e-3*np.ones(num_periods*num_parameters), args=(station_idf, getattr(self, IDF_type)))
            synth_IDF = self.construct_multi_IDFs(station_idf.index, Duracion_lluvia, ajuste.x, getattr(self, IDF_type))
            IDF_curve_fit = synth_IDF.T

        elif method == 'potential_regression':

            table = self.Gages_idf[station].T.sort_index(ascending=False)
            table['min'] = table.index*60
            # Función auxiliar para calcular d y n para un periodo específico
            def calcular_dn(x, y):
                n = len(x)
                ln_x = [math.log(xi) for xi in x]
                ln_y = [math.log(yi) for yi in y]
                sum_ln_x = sum(ln_x)
                sum_ln_y = sum(ln_y)
                sum_ln_x_ln_y = sum(lnxi * lnyi for lnxi, lnyi in zip(ln_x, ln_y))
                sum_ln_x_squared = sum(lnxi**2 for lnxi in ln_x)
                
                numerador = (n * sum_ln_x_ln_y) - (sum_ln_x * sum_ln_y)
                denominador = (n * sum_ln_x_squared) - (sum_ln_x**2)
                n_value = numerador / denominador
                ln_d = (sum_ln_y - n_value * sum_ln_x) / n
                d_value = math.exp(ln_d)
                
                return d_value, n_value

            # Preparar el DataFrame de resultados
            resultados = []

            # Calcular d y n para cada periodo
            for periodo in self.Return_period:
                x = table['min'].values
                y = table[periodo].values
                d, n = calcular_dn(x, y)
                resultados.append({
                    'Return_period': int(periodo),
                    'regression_coef_[d]': d,
                    'regression_coef_[n]': n
                })

            # Crear DataFrame de resultados
            df_resultados = pd.DataFrame(resultados)
            
            # Calcular promedios
            promedios = df_resultados[['regression_coef_[d]', 'regression_coef_[n]']].mean()
            promedios_df = pd.DataFrame(promedios).T
            promedios_df['Return_period'] = 'Promedio'
            
            # Combinar resultados y promedios
            df_summary = pd.concat([df_resultados, promedios_df]).reset_index(drop=True)

            d, e = calcular_dn(df_summary['Return_period'][:-1], df_summary['regression_coef_[d]'][:-1])
            b = df_summary['regression_coef_[n]'][:-1].mean()*-1
            durations = np.round(Duracion_lluvia*60, 4)

            # Create an empty DataFrame with frequencies as index and durations as columns
            intensity_df = pd.DataFrame(index=self.Return_period, columns=durations)

            # Calculate the intensities and fill the DataFrame
            for T in self.Return_period:
                for t in durations:
                    intensity_df.loc[T, t] = getattr(self, 'IDF_typeV_pot_reg')(T, t, d, e, b)

            intensity_df = intensity_df.T
            intensity_df.index = intensity_df.index/60
            
            IDF_curve_fit = intensity_df.copy()
            
        else:
            raise ValueError("Method must be 'curve_fit' or 'least_squares' or 'potential_regression")
        
        if plot:
            # Create plot of the results
            fig, ax = plt.subplots(figsize=(10, 6))
            colores = plt.cm.viridis(np.linspace(0, 1, len(IDF_curve_fit.columns)))
            
            for idx, tr in enumerate(self.Return_period):
                co = station_idf.loc[tr, :]
                cs = IDF_curve_fit[tr]
                ax.scatter(self.Durations, co.values, color=colores[idx], label=f'T = {tr} years (obs)', s=30, alpha=0.7)
                ax.plot(Duracion_lluvia, cs, color=colores[idx], linewidth=2)
            
            ax.set_xlim(0.0833333, self.Durations.max())
            ax.set_xlabel('Duration (h)', fontsize=14)
            ax.set_ylabel('Intensity (mm/h)', fontsize=14)
            ax.set_title(f'IDF Curves - Station {station}', fontsize=16)
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            
            return IDF_curve_fit, fig
        
        return IDF_curve_fit
    
    def __calculate_goodness_of_fit(self, station, IDF_type=None, method=None):
        """
        Calculate goodness of fit metrics for the IDF curves, including a global fit.
        
        Args:
            station (str): Station to analyze.
            IDF_type (str): Type of IDF equation to use.
            method (str): Fitting method to use ('curve_fit' or 'least_squares').
        
        Returns:
            pd.DataFrame: DataFrame containing R², RMSE, and MAE for each return period and global fit.
        """
        if method is None:
            method = self.method
        if IDF_type is None:
            IDF_type = self.IDF_type
        
        station_idf = self.Gages_idf[station]
        
        # Fit IDF curves
        IDF_curve_fit = self.IDF_fit(station, IDF_type, method, plot=False)
        
        # Prepare data for global fit calculation
        observed_all = []
        predicted_all = []
        
        # Calculate metrics
        metrics = {}
        for tr in self.Return_period:
            observed = station_idf.loc[tr, :]
            predicted = IDF_curve_fit[tr].loc[self.Durations]
            
            observed_all.extend(observed)
            predicted_all.extend(predicted)
            
            r2 = r2_score(observed, predicted)
            rmse = np.sqrt(mean_squared_error(observed, predicted))
            mae = mean_absolute_error(observed, predicted)
            
            metrics[tr] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            }
        
        # Calculate global fit
        global_r2 = r2_score(observed_all, predicted_all)
        global_rmse = np.sqrt(mean_squared_error(observed_all, predicted_all))
        global_mae = mean_absolute_error(observed_all, predicted_all)
        
        metrics['Global'] = {
            'R2': global_r2,
            'RMSE': global_rmse,
            'MAE': global_mae
        }
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.index.name = 'Return_Period'
        
        return metrics_df
    
    def goodness_of_fit(self, station, method='all'):
        """
        Compare the goodness of fit for different IDF types using curve_fit and least_squares methods.

        Args:
            station (str): Station to analyze.
            method (str): Method to use for comparison. Options are 'curve_fit', 'least_squares', 'all'.

        Returns:
            pd.DataFrame: DataFrame containing R², RMSE, and MAE for each IDF type and method.
        """
        IDF_types = ['IDF_typeI', 'IDF_typeII', 'IDF_typeIII', 'IDF_typeIV', 'IDF_typeV']
        results_goodness_curve_fit = pd.DataFrame(index=IDF_types, columns=['R2', 'RMSE', 'MAE'])
        results_goodness_least_squares = pd.DataFrame(index=IDF_types, columns=['R2', 'RMSE', 'MAE'])

        if method in ['curve_fit', 'all']:
            for IDF_type in IDF_types:
                curve_fit_results = self.__calculate_goodness_of_fit(station, IDF_type=IDF_type, method='curve_fit').loc['Global'].values
                results_goodness_curve_fit.loc[IDF_type] = curve_fit_results
            results_goodness_curve_fit['method'] = 'curve_fit'

        if method in ['least_squares', 'all']:
            for IDF_type in IDF_types:
                least_squares_results = self.__calculate_goodness_of_fit(station, IDF_type=IDF_type, method='least_squares').loc['Global'].values
                results_goodness_least_squares.loc[IDF_type] = least_squares_results
            results_goodness_least_squares['method'] = 'least_squares'

        if method == 'curve_fit':
            results_goodness = results_goodness_curve_fit.copy()
        elif method == 'least_squares':
            results_goodness = results_goodness_least_squares.copy()
        elif method == 'all':
            results_goodness = pd.concat([results_goodness_curve_fit, results_goodness_least_squares])
            results_goodness.sort_values(by='R2', ascending=False, inplace=True)
        else:
            raise ValueError("Method must be 'curve_fit', 'least_squares', or 'all'")

        return results_goodness