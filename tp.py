# %%
# ----- load data sets -----
import my_toolbox as mtb
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.getcwd())

# %%
# Get data from source

data_sets = [
    # first address records hosted on git
    'https://raw.githubusercontent.com/IshMakwana/gwu_time_series_tp/main/pollution_record1.csv',
    # whole dataset hosted on google drive
    'https://drive.google.com/u/2/uc?id=1_RqOW68TZu2gPQtbWa6TVThEsNRUzRiR&export=download']

# df = pd.read_csv(data_sets[0], parse_dates=['Date'])
pollution_df = pd.read_csv(data_sets[1], parse_dates=['Date'])

# I am filtering data by unique addresses, with more than 5000 observations and then doing analysis on the filtered data


def findIthAddress(ith=1):
    i = 0
    for addy in pollution_df['Address'].unique():
        records = pollution_df.query('Address == @addy')
        if len(records) > 5000:
            # records.to_csv('{0}pollution_record{1}.csv'.format(write_dir, i+1))
            # I wrote csv files with records and uploaded them to git for easier access
            # the draw back of loading the whole data is that it takes 5-6 seconds to read_csv with a large hosted file.
            if (i+1) == ith:
                return records
            i += 1


df = findIthAddress()
df.reset_index(inplace=True)
# df.set_index('Date', inplace=True)

print(pollution_df.shape)
print(df.shape)

# %% [markdown]
# Things to include in the report:
# - Cover page
# - Table of contents
# - Table of figures and tables
# - Abstract
# - Introduction: An overview of the time series analysis and modeling process and an outline of the report.

# %% [markdown]
# 6. Description of the dataset.
# - Describe the independent variable(s) and dependent variable

# %%
# The dependent variable for this data set is 'O3 AQI'
# Other numerical variables will be considered the independent variables.
dep_var = ['O3 AQI']
ind_var = [
    'O3 Mean', 'O3 1st Max Value', 'O3 1st Max Hour',
    'CO Mean', 'CO 1st Max Value', 'CO 1st Max Hour', 'CO AQI',
    'SO2 Mean', 'SO2 1st Max Value', 'SO2 1st Max Hour', 'SO2 AQI',
    'NO2 Mean', 'NO2 1st Max Value', 'NO2 1st Max Hour', 'NO2 AQI'
]

# The following are the categorical variables for this data
cat_var = ['Address',
           'State',
           'County',
           'City']

df = df[['Date'] + cat_var + dep_var + ind_var]


def printList(lst):
    for l in lst:
        print(f'{l}')


print('----- Question 6 Solution ----- ')
print(f'Dependent Variable (1): ')
printList(dep_var)
print(f'Independent Variables ({len(ind_var)}):')
printList(ind_var)
print(f'Categorical Variables ({len(cat_var)}):')
printList(cat_var)

# %% [markdown]
# - a. Pre-processing dataset: Dataset cleaning for missing observation. You must follow the data cleaning techniques for time series dataset.
# - b. May need to perform one-hot-encoding to convert categorical feature to a numericalfeature.
# - c. May need down sampling or up sampling for the time series qualification.
# - d. The plot of the dependent variable versus time. Write down your observations.
# - e. ACF/PACF of the dependent variable. Write down your observations.
# - f. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient. Write down your observations.
# - g. Split the dataset into train set (80%) and test set (20%)

# %%
# ----- 6.a -----


def cleanAndFill():
    print('----- Question 6(a) Solution ----- ')
    # print('Let\'s check if there are missing observations in our dataset...')

    # First we will check if there are any empty observations:
    print(f'Check missing observations \n{df.isna().sum()}')
    # Looking at the above, we can say there are no empty observations.

    # Now we can check if all dates are included in the range
    # Create a full date range and match with out dataset
    dates = df['Date']
    dr = pd.date_range(start=df['Date'][0],
                       end=df['Date'][len(df)-1], freq='d')
    print(f'Start Date: {dates[0]}, End Date: {dates[len(df)-1]}')
    print(f'Date Range: {len(dr)}, Dataset: {len(dates)}')

    # We have some dates not present in the dataset, lets find out where
    d1 = list(dr)
    d2 = list(dates)
    # We have some dates not present in the dataset, lets find out where
    missing_dates = [x for x in d1 if x not in d2]
    # print(f'length: {missing_dates}')

    # We can fill the missing_dates using drift method for the numerical data, but we can use the naive method for categorical data.
    cleaned = mtb.driftFill(df, 'Date', dr, dep_var + ind_var)

    dates = cleaned['Date']
    print(f'Date Range: {len(dr)}, Drift Filled Dataset: {len(dates)}')

    # Find out some basic stats for the dep var (max, min, mean, vari, std)
    print(f"""
    Following are some stats for the O3 AQI
        Max:        {np.max(cleaned[dep_var])}
        Min:        {np.min(cleaned[dep_var])}
        Mean:       {np.mean(cleaned[dep_var]):.3f}
        Variance:   {np.var(cleaned[dep_var[0]]):.3f}
        St. Dev.:   {np.std(cleaned[dep_var[0]]):.3f}
    """)

    # ----- 6b -----
    print('----- Question 6(b) Solution ----- ')
    for c in cat_var:
        print(df[c].unique())

    return cleaned


df = cleanAndFill()


# %%
def downSampleAndStuff():
    # The above tells us that all categorical variables are constant, so we can avoid encoding them and just ignore for this dataset.

    # ----- 6c -----
    print('----- Question 6(c) Solution ----- ')
    w_df = mtb.downSample(7, df, 'Date', dep_var + ind_var)
    m_df = mtb.downSample(30, df, 'Date', dep_var + ind_var)

    print(f'\nThe # of observations for weekly sampled data: {len(w_df)}')

    mtb.plotXY(w_df['Date'], w_df[dep_var], 'Weekly Sampled Data', 'Timeline', dep_var[0], fs=(9, 5))
    # plotXY(m_df['Date'], m_df[dep_var], dep_var[0], 'Date', 'Monthly Sampled Data', fs=(9,3))

    # we will use the weekly sampled data

    # ----- 6d -----
    print('----- Question 6(d) Solution ----- ')
    """
    This plot is for the unfilled dataset. 
    Looking at this data vs the drift filled data, don't see any difference. 
    For the sake of having a uniform date range, we will use the drift filled data. 

    plotXY(df['Date'], df[dep_var], dep_var[0], 'Date', 'Original Data')
    """
    s = 365

    mtb.plotXY(df['Date'], df[dep_var], 'Drift Filled Data',
               'Timeline', dep_var[0], fs=(9, 5))

    di_df = df.set_index('Date')
    ds_df = mtb.deSeas(di_df[dep_var[0]], s)

    mtb.plotXY(df['Date'][s:], ds_df, 'Deseasonalized Data',
               'Timeline', dep_var[0], fs=(9, 5))

    # ds_df = mtb.deSeas(ds_df, s)

    # mtb.plotXY(range(len(ds_df)), ds_df, 'Deseasonalized Data x2', 'Timeline', dep_var[0], fs=(9, 5))
    # uds_df = mtb.unDeSeas(ds_df, s)
    # mtb.plotXY(uds_df, uds_df, 'Reseasonalized Data', 'Timeline', dep_var[0], fs=(9, 5))

    # Observation
    # -> Looking at the plot below, I can say that the data is not really trending but highly seasonal (in a yearly fashion).
    return ds_df, w_df


ds_df, w_df = downSampleAndStuff()

# %%

mtb.ACF_PACF_Plot(df[dep_var[0]], 80)

# %%
# ----- 6f -----


def correlationMap():
    correlation_matrix = df[dep_var + ind_var].corr()

    # Creating a heatmap to visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0, annot=True,
                cmap=sns.diverging_palette(20, 220, n=200), linewidths=0.5)
    plt.title("Correlation Matrix of Dependent and Independent Variables")
    plt.show()
    # Some other O3 variables (O3 Mean and O3 1st Max Value) seem to be correlated to O3 AQI, but others are highly uncorrelated with O3 AQI.


correlationMap()

# %%


def splitData():
    # ----- 6g -----
    print('----- Question 6(g) Solution ----- ')
    # maybe standardize data
    train, test = train_test_split(
        df[['Date'] + dep_var + ind_var], shuffle=False, test_size=0.2)
    print(f'Train set shape: {train.shape} and Test set shape: {test.shape}')

    return train, test


train, test = splitData()

# %% [markdown]
# 7. Stationarity [3pts]:
#
# - Check for a need to make the dependent variable stationary.
# - If the dependent variable is not stationary, you need to use the techniques discussed in class to make it stationary.
# - Perform ACF/PACF analysis for stationarity. You need to perform ADF-test & KPSS-test and plot the rolling mean and variance for the raw data and the transformed data.
# - Write down your observations.

# %%
# Rolling Mean and Variance plot
di_df = df.set_index('Date')
mtb.plot_rolling_mean_var(di_df[dep_var[0]], 'Timeline', dep_var[0])
print('----- Question 7 Solution ----- ')

# Looking at the Rolling Mean and Variance plots of O3 AQI, rolling mean is stabilizing over time, slightly increasing,
# and the rolling variance stabilizes but is decreasing and may need some transformation, but let's see what we can find from ADF and KPSS tests

# %%


def showAdfKpss():
    # ADF and KPSS Tests
    print(f'ADF Test of {dep_var[0]}')
    mtb.printAdf(df[dep_var])
    print(f'\nKPSS Test of {dep_var[0]}')
    mtb.printKpss(df[dep_var])


    # Looking at these test results, we can confidently say that the dependent variable is stationary and we don't need to further transform the data.
print('----- Question 7 Solution ----- ')
showAdfKpss()

# %% [markdown]
# 8 Time series Decomposition[4pts]:
# - Approximate the trend and the seasonality and plot the detrended and the seasonally adjusted data set using STL method.
# - Find out the strength of the trend and seasonality.
# - Refer to the lecture notes for distinct types of time series decomposition techniques.

# %%
# ----- 8 -----
mtb.STLDecompose(df, dep_var[0], df['Date'], 365,
                 f'{dep_var[0]} Decomposition')
# The measurements here are strange because the data is barely trended and highly seasonl

# %% [markdown]
# 9 Holt-Winters method[3pts]:
# - Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set.

# %%

xlabel = 'Timeline'
ylabel = dep_var[0]

def printErrComparison(comparison):
    err_df = pd.DataFrame(comparison, columns=mtb.ERR_HEAD)
    print(err_df)

hw_comparison = []

def showHoltWinters(train, test):
    yf = pd.DataFrame(train[dep_var[0]]).set_index(train['Date'])
    yt = pd.DataFrame(test[dep_var[0]]).set_index(test['Date'])

    pred = mtb.plotHoltWinters(yf, yt,
                               trend='add',
                               seasonal='add',
                               damped=True,
                               seasonal_periods=365, title='Holts Winters (add) S + T', xlabel=xlabel, ylabel=ylabel)
    hw_comparison.append(mtb.getAccuracy(
        yt.values, pred.values, 'Holts Winters (add) S + T'))
    pred = mtb.plotHoltWinters(yf, yt,
                               # trend='mul',
                               seasonal='add',
                               # damped=True,
                               seasonal_periods=365, title='Holts Winters (add) S', xlabel=xlabel, ylabel=ylabel)
    hw_comparison.append(mtb.getAccuracy(
        yt.values, pred.values, 'Holts Winters (add) S'))
    pred = mtb.plotHoltWinters(yf, yt,
                               trend='mul',
                               seasonal='mul',
                               damped=True,
                               seasonal_periods=365, title='Holts Winters (mul) S + T', xlabel=xlabel, ylabel=ylabel)
    hw_comparison.append(mtb.getAccuracy(
        yt.values, pred.values, 'Holts Winters (mul) S + T'))
    pred = mtb.plotHoltWinters(yf, yt,
                               # trend='mul',
                               seasonal='mul',
                               # damped=True,
                               seasonal_periods=365, title='Holts Winters (mul) S', xlabel=xlabel, ylabel=ylabel)
    hw_comparison.append(mtb.getAccuracy(
        yt.values, pred.values, 'Holts Winters (mul) S'))

    # mtb.plotTable(hw_comparison, mtb.ERR_HEAD, 'Holt Winters Comparison')

    # After some trial and error, I have found that additive seasonality works best for this data,
    # and looking at the data, we can tell that the seasonal period is a year. As we have daily data,
    # I have chosen the seasonal period = 365 (for some reason 364 works better)
    printErrComparison(hw_comparison)


# wdf = mtb.downSample(7, df, 'Date', dep_var + ind_var)
# mtb.plot_rolling_mean_var(wdf[dep_var[0]])
# train, test = train_test_split(df, test_size=0.2, shuffle=False)
showHoltWinters(train, test)

# %% [markdown]
# 10 Feature selection/dimensionality reduction: [under multiple linear regression]
# - You need to have a section in your report that explains how the feature selection was performed and whether the collinearity exists not.
# - PCA, backward stepwise regression, SVD decomposition, condition number and VIF analysis are needed.
# - You must explain which feature(s) need to be eliminated and why.

# %%
# I am not sure if we have done PCA during the class, labs or home works. Pending for now...
mtb.svdDecompose(df, ind_var)

# %%


def ldiff(xcl, all):
    return [x for x in all if x not in xcl]


def showFeatireSelect():
    # Backward Step Regression

    std_df = mtb.standardize(df[dep_var + ind_var])
    stdf, stdt = train_test_split(std_df, test_size=0.2, shuffle=False)

    # Perform backward step regression but maximize Adjusted R2
    adj_rem = mtb.backStepReg(stdf, dep_var[0], ind_var)
    print('\nFeatures to be removed: ')
    printList(adj_rem)
    print('Recommended features: ')
    adj_rec = ldiff(adj_rem, ind_var)
    printList(adj_rec)
    print('\n')

    # Perform backward step regression remove all features where P-Value > 0.05
    pval_rem = mtb.backStepReg(stdf, dep_var[0], ind_var, use_adjr2=False)
    print('\nFeatures to be removed: ')
    printList(pval_rem)
    print('Recommended features: ')
    pval_rec = ldiff(pval_rem, ind_var)
    printList(pval_rec)
    print('\n')

    # Perform backward step regression remove all features where VIF > 10
    vif_rem = mtb.vifMethod(stdf, dep_var[0], ind_var)
    print('\nFeatures to be removed: ')
    printList(vif_rem)
    print('Recommended features: ')
    vif_rec = ldiff(vif_rem, ind_var)
    printList(vif_rec)

    return adj_rec, pval_rec, vif_rec


arec, prec, vrec = showFeatireSelect()
# %%

lr_comparison = []


def linRegCompare():
    train, test = train_test_split(df.set_index(
        'Date'), test_size=0.2, shuffle=False)
    # print(regress_features)

    method = 'Adjusted R2'
    predict = mtb.plotRegressPrediction(
        train, test, dep_var[0], arec, method, xlabel=xlabel, ylabel=ylabel)
    lr_comparison.append(mtb.getAccuracy(
        test[dep_var[0]].values, predict.values, method))

    method = 'P-Value'
    predict = mtb.plotRegressPrediction(
        train, test, dep_var[0], prec, method, xlabel=xlabel, ylabel=ylabel)
    lr_comparison.append(mtb.getAccuracy(
        test[dep_var[0]].values, predict.values, method))

    method = 'VIF'
    predict = mtb.plotRegressPrediction(
        train, test, dep_var[0], vrec, method, xlabel=xlabel, ylabel=ylabel)
    lr_comparison.append(mtb.getAccuracy(
        test[dep_var[0]].values, predict.values, method))

    printErrComparison(lr_comparison)


linRegCompare()

# %% [markdown]
# 11 Base-models[3pts]: average, naïve, drift, simple and exponential smoothing.
# - You need to perform an h-step prediction based on the base models and compare the SARIMA model performance with the base model predication.

# %%

bs_compparison = []
def showBasicModels():
    train, test = train_test_split(df.set_index(
        'Date'), test_size=0.2, shuffle=False)
    # Plot Average, Naive, Drift and SES models, using monthly data for better charts
    trn = train[dep_var[0]]
    tst = test[dep_var[0]]

    method = 'Average'
    frcst, _ = mtb.getAvgModel(trn.values, tst.values)
    mtb.plotForecast(trn, tst, frcst, method, xlabel=xlabel, ylabel=ylabel)
    bs_compparison.append(mtb.getAccuracy(tst.values, frcst, method))

    method = 'Naive'
    frcst, _ = mtb.getNaiveModel(trn.values, tst.values)
    mtb.plotForecast(trn, tst, frcst, method, xlabel=xlabel, ylabel=ylabel)
    bs_compparison.append(mtb.getAccuracy(tst.values, frcst, method))

    method = 'Drift'
    frcst, _ = mtb.getDriftModel(trn.values, tst.values)
    mtb.plotForecast(trn, tst, frcst, method, xlabel=xlabel, ylabel=ylabel)
    bs_compparison.append(mtb.getAccuracy(tst.values, frcst, method))

    alpha = 0.3
    method = f'SES a={alpha}'
    frcst, _ = mtb.getSESModel(trn.values, tst.values, alpha)
    mtb.plotForecast(trn, tst, frcst, method, xlabel=xlabel, ylabel=ylabel)
    bs_compparison.append(mtb.getAccuracy(tst.values, frcst, method))

    printErrComparison(bs_compparison)

showBasicModels()

# %% [markdown]
# 12. Develop the multiple linear regression model that represents the dataset. Check the accuracy of the developed model. [10pts]
# - a. You need to include the complete regression analysis in your report. Perform one-step ahead prediction and compare the performance versus the test set.
# - b. Hypothesis tests analysis: F-test, t-test.
# - c. Cross validation for time series.
# - d. Display MSE, AIC, BIC, RMSE, R-squared and Adjusted R-squared
# - e. ACF of residuals.
# - f. Q-value
# - g. Variance and mean of the residuals.
# - h. Plot the train, test, and predicted values in one plot.

# %%


def multipleLinRegModels(data, feats, method):
    # ----- 12.a. -----
    # regress_remove = ['NO2 AQI', 'CO Mean']

    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    # print(regress_features)

    model, predict = mtb.plotTestVPrediction(train, test, dep_var[0], feats,
                                             f'{method} 1-Step', xlabel=xlabel, ylabel=ylabel)
    print('----- Question 12(a) Solution ----- ')
    print(model.summary())

    e = test[dep_var[0]].values.flatten() - predict.values.flatten()

    # ----- 12.b. -----
    print('----- Question 12(b) Solution ----- ')
    mtb.f_t_test(model, f'{method} model F,T-tests')
    # ----- 12.c. -----
    # cross validation of time series
    print('----- Question 12(c) Solution ----- ')
    print('Cross Validation - 5 Splits')
    mtb.cross_validation_linreg(data, dep_var[0], feats)

    # ----- 12.d. -----
    print('----- Question 12(d) Solution ----- ')
    mtb.showRegressionMetrics(model, e.flatten(), method)

    lags = 200
    # ----- 12.e. -----
    acf = mtb.plotAcf(e.flatten(), num_lags=lags, title=f'{method} Residual')
    # ----- 12.f. -----
    # q - value vs q-critical
    print('----- Question 12(f) Solution ----- ')
    mtb.residual_analysis(acf, lags, len(data), f'{method} Residual Analysis')

    # ----- 12.g. -----
    print('----- Question 12(g) Solution ----- ')
    mtb.m_v(e.flatten(), f'{method} Residual Mean and Variance')

    # plotTestVPrediction(mtrain, mtest, 'Prediction w/ Monthly Data')

    # ----- 12.h. -----
    # Based on step 10, we choose the following, by maximizing the Adjusted R2

    # Plot regression prediction for daily, weekly and monthly data
    _ = mtb.plotRegressPrediction(
        train, test, dep_var[0], feats, f'{method} method Forecast', xlabel=xlabel, ylabel=ylabel)
    # plotRegressPrediction(mtrain, mtest, 'Prediction w/ Monthly Data')


# feats = prec
# method = 'P-Value'
# feats = vrec
# method = 'VIF'
feats = arec
method = 'Adjusted R2'
multipleLinRegModels(df.set_index('Date'), feats, method)

# %%

wi_df = w_df.set_index('Date')
train, test = train_test_split(wi_df[dep_var[0]], test_size=0.2, shuffle=False)

arima_model_a = mtb.predictARIMA(
    train, order=(0, 0, 0), trend='t', seasonal=(1, 0, 0, 52))
arima_model_b = mtb.predictARIMA(
    train, order=(0, 0, 0), trend='t', seasonal=(1, 0, 2, 52))

#%%

ar_comparison = []
def arima_plot(model, train, test, method):
    model_hat = model.forecast(len(test))
    plt.figure(figsize=(12, 4))
    plt.title('Weekly Data Test v Forecast')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Test Data')
    plt.plot(model_hat, label='ARIMA Forecast')
    ar_comparison.append(mtb.getAccuracy(
        test.values, model_hat.values, method))
    plt.legend()
    plt.grid()
    plt.show()    
    return model_hat

_ = arima_plot(arima_model_a, train, test, 'ARIMA (1, 0, 0, 52)')
model_b_predict = arima_plot(arima_model_b, train, test, 'ARIMA (1, 0, 2, 52)')
printErrComparison(ar_comparison)

mtb.ACF_PACF_Plot(wi_df[dep_var[0]].values.flatten(), 20)

print('----- Question 14 Solution ----- ')
mtb.printLM(wi_df[dep_var[0]], 1, 0)
mtb.printLM(wi_df[dep_var[0]], 1, 2)

# %%
lags = 25

print('----- Question 16(a) Solution ----- ')
# print(len(arima_model_b.resid.values.flatten()))
racf = mtb.acf(arima_model_b.resid.values.flatten(), lags)
mtb.residual_analysis(racf, lags, len(train), 'ARIMA (1, 0, 2, 52)')

print('----- Question 16(c) Solution ----- ')
print(f'Bias: {np.mean(arima_model_b.resid.values.flatten()):.3f}')

# %%
print('----- Question 16(d) Solution ----- ')
print(f'Variance of Residuals: {np.var(arima_model_b.resid.values.flatten()):.3f}')
print(f'Variance of Forecast Errors: {np.var(test.values.flatten() - model_b_predict.values.flatten()):.3f}')

# %%

print('----- Question 16(e) Solution ----- ')
ar_params = [-0.994]
ma_params = [0.382,-0.068]

print(f'AR Params; {ar_params}, MA Params: {ma_params}')
print(f'AR Roots: {mtb.roots(ar_params)}')
print(f'MA Roots: {mtb.roots(ma_params)}')

# %%

print('----- Question 17 Solution ----- ')
printErrComparison(bs_compparison + hw_comparison + lr_comparison + ar_comparison)

# %%
# print('----- Question 19 Solution ----- ')

feats = arec
method = 'Adjusted R2'
train, test = train_test_split(df.set_index('Date'), test_size=0.2, shuffle=False)
_,_ = mtb.plotTestVPrediction(
        train, test, dep_var[0], feats, f'{method} LSE Forecast', xlabel=xlabel, ylabel=ylabel)
# %%
