# Time Series Analysis with ARIMA, SARIMA, MA, VARMA, and DeepAnT

This repository provides a comprehensive analysis of time series data using various statistical models such as ARIMA, SARIMA, MA, and VARMA, alongside the deep learning architecture DeepAnT. The project focuses on modeling and forecasting time series data, with DeepAnT applied to anomaly detection in Canadian climate data.

## Project Overview

The goal of this project is to explore different methodologies for time series analysis and forecast accuracy. Traditional models like ARIMA, SARIMA, MA, and VARMA are employed for their robustness in time series forecasting, while DeepAnT is utilized to uncover complex patterns and anomalies.

## Methodology

### ARIMA (AutoRegressive Integrated Moving Average)
ARIMA combines autoregressive (AR), differencing (I), and moving average (MA) components.

**Mathematical Equation:**
$$y_t = c + \epsilon_t + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-j}$$

- **$y_t$**: The actual value at time $t$.
- **$c$**: Constant term.
- **$p$**: Number of lag observations (autoregressive part).
- **$q$**: Size of the moving average window.
- **$\phi_i$**: Coefficients for the lagged observations.
- **$\theta_j$**: Coefficients for the lagged forecast errors.
- **$\epsilon_t$**: Error term.

### SARIMA (Seasonal ARIMA)
SARIMA extends ARIMA to include seasonal components.

**Mathematical Equation:**
$$y_t = c + \epsilon_t + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-j} + \sum_{i=1}^{P}\Phi_i Y_{t-i} + \sum_{j=1}^{Q}\Theta_j E_{t-j}$$

- **$Y_{t-i}$**: Seasonal autoregressive terms.
- **$E_{t-j}$**: Seasonal error terms.
- **$P$**: Number of seasonal autoregressive terms.
- **$Q$**: Number of seasonal error terms.
- **$\Phi_i$**, **$\Theta_j$**: Seasonal coefficients.

### MA (Moving Average)
MA models account for the dependency between an observation and a residual error from a moving average model.

**Mathematical Equation:**
$$y_t = \mu + \epsilon_t + \sum_{j=1}^{q}\theta_j \epsilon_{t-j}$$

- **$\mu$**: Mean of the series.
- **$q$**: Size of the moving average window.
- **$\theta_j$**: Coefficients for the lagged forecast errors.
- **$\epsilon_t$**: Error term.

### VARMA (Vector AutoRegressive Moving Average)
VARMA is used for multivariate time series data, capturing relationships between multiple time series.

**VAR Part:**
$$y_t = c + \sum_{i=1}^{p}\Phi_i y_{t-i} + \epsilon_t$$

- **$y_t$**: Vector of time series values.
- **$\Phi_i$**: Coefficient matrices.
- **$p$**: Number of lag observations.
- **$\epsilon_t$**: Vector of error terms.

**MA Part:**
$$y_t = \mu + \sum_{j=1}^{q}\Theta_j \epsilon_{t-j} + \epsilon_t$$

- **$\mu$**: Mean vector.
- **$q$**: Size of the moving average window.
- **$\Theta_j$**: Coefficient matrices for lagged errors.
- **$\epsilon_t$**: Error term.

### DeepAnT Architecture
DeepAnT (Deep Anomaly Detection) is a deep learning model designed for anomaly detection in time series data. The model consists of a Convolutional Neural Network (CNN) followed by `maxpooling`, `dropout` and `dense` layers.

#### Application to Canadian Climate Data
DeepAnT was applied to detect anomalies in Canadian climate data. The model was trained to learn normal patterns in the data, and deviations from these patterns were flagged as anomalies. This approach is effective in identifying unusual climate events and trends.

## Acknowledgements

- **DeepAnT Model Implementation:** The implementation of the DeepAnT architecture is adapted from the [DatascienceArticle](https://towardsdatascience.com/deepant-unsupervised-anomaly-detection-for-time-series-97c5308546ea).
