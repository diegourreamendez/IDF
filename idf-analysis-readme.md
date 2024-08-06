# IDF Analysis Tool

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Class: IDFAnalysis](#class-idfanalysis)
5. [Methods](#methods)
6. [Examples](#examples)
7. [Visualizations](#visualizations)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

The IDF Analysis Tool is a Python-based solution for performing Intensity-Duration-Frequency (IDF) analysis on rainfall data. This tool encapsulates methods for calculating annual maximum intensities, fitting statistical models, generating IDF curves, and plotting results.

IDF analysis is crucial in hydrological studies and water resource management, providing essential information for the design of drainage systems, flood control structures, and other water-related infrastructure.

## Installation

To use the IDF Analysis Tool, you need to have Python installed on your system. Additionally, you'll need to install the following dependencies:

```bash
pip install pandas numpy scipy matplotlib fitter statsmodels
```

You'll also need to ensure you have the `Julia_Genextreme` module available in your Python environment.

## Usage

To use the IDF Analysis Tool, you need to import the `IDFAnalysis` class from the module:

```python
from idf_analysis import IDFAnalysis
```

## Class: IDFAnalysis

The `IDFAnalysis` class is the core of this tool. It takes the following parameters during initialization:

- `historic_hourly`: A pandas DataFrame containing historical hourly rainfall data.
- `Durations`: A numpy array of durations to analyze (in hours).
- `Return_periods`: A numpy array of return periods to calculate.
- `distribution`: The statistical distribution to use for fitting (default is 'genextreme').
- `model`: The model engine to use for fitting ('scipy_stats' or 'Julia_stats', default is 'scipy_stats').
- `method`: The method for fitting IDF curves ('curve_fit' or 'least_squares', default is 'curve_fit').
- `IDF_type`: The type of IDF equation to use (default is 'IDF_typeI').

## Methods

### 1. `_calculate_intensity_annual_max()`

This method calculates annual maximum intensities for each duration and station.

### 2. `_fit_models()`

This method fits statistical models to annual maximum intensities.

### 3. `_calculate_idf()`

This method calculates Intensity-Duration-Frequency (IDF) values for each station.

### 4. `get_idf_table(station=None)`

This method returns the IDF table for a specific station or all stations.

### 5. `plot_cdf_models(station)`

This method generates Cumulative Distribution Function (CDF) plots for a specific station.

### 6. `plot_qq_models(station)`

This method generates Quantile-Quantile (Q-Q) plots for a specific station.

### 7. `IDF_fit(station, IDF_type=None, method=None, plot=True)`

This method fits the IDF curve for a specific station using the specified method.

## Examples

Let's walk through a step-by-step example of how to use the IDF Analysis Tool:

### Step 1: Prepare Your Data

First, you need to prepare your historical hourly rainfall data in a pandas DataFrame format. Each column should represent a station, and the index should be the datetime.

```python
import pandas as pd
import numpy as np

# Load your data (replace with your actual data loading method)
historic_hourly = pd.read_csv('your_rainfall_data.csv', index_col=0, parse_dates=True)

# Define durations and return periods
durations = np.array([1, 2, 3, 6, 12, 24])  # in hours
return_periods = np.array([2, 5, 10, 25, 50, 100])  # in years
```

### Step 2: Initialize the IDFAnalysis Class

```python
idf_analysis = IDFAnalysis(historic_hourly, durations, return_periods)
```

### Step 3: Get IDF Table for a Specific Station

```python
station_name = 'Station1'
idf_table = idf_analysis.get_idf_table(station_name)
print(idf_table)
```

### Step 4: Plot CDF Models

```python
cdf_plot = idf_analysis.plot_cdf_models(station_name)
cdf_plot.savefig('cdf_plot.png')
```

[Insert CDF Plot Image Here]

### Step 5: Plot Q-Q Models

```python
qq_plot = idf_analysis.plot_qq_models(station_name)
qq_plot.savefig('qq_plot.png')
```

[Insert Q-Q Plot Image Here]

### Step 6: Fit IDF Curves

```python
idf_curve, idf_plot = idf_analysis.IDF_fit(station_name)
idf_plot.savefig('idf_curve.png')
```

[Insert IDF Curve Plot Image Here]

## Visualizations

The IDF Analysis Tool provides several types of visualizations to help interpret the results:

1. **CDF Plots**: These plots show the cumulative distribution function of the fitted model compared to the observed data for each duration.

2. **Q-Q Plots**: These plots compare the quantiles of the theoretical distribution to the quantiles of the observed data, helping to assess the goodness of fit.

3. **IDF Curves**: These plots show the fitted IDF curves along with the observed data points, allowing for visual comparison of the model fit to the data.

## Contributing

Contributions to the IDF Analysis Tool are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License

[Insert your chosen license information here]

---

This README provides a comprehensive guide to using the IDF Analysis Tool. For more detailed information about specific methods or advanced usage, please refer to the inline documentation in the source code.
