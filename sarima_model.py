import eia
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm_api
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def retrieve_time_series(api, series_ID):
    """
    Return the time series dataframe, based on API and unique Series ID
    """
    #Retrieve Data By Series ID 
    series_search = api.data_by_series(series=series_ID)
    ##Create a pandas dataframe from the retrieved time series
    df = pd.DataFrame(series_search)
    return df

def plot_data(df, x_variable, y_variable, title):
    """
    Plot the x- and y- variables against each other, where the variables are columns in
    a pandas dataframe
    Args:
        df: Pandas dataframe. 
        x_variable: String. Name of x-variable column
        y_variable: String. Name of y-variable column
        title: String. Desired title name
    """
    fig, ax = plt.subplots()
    ax.plot_date(df[x_variable], 
                 df[y_variable], marker='', linestyle='-', label=y_variable)
    fig.autofmt_xdate()
    plt.title(title)
    plt.show()

def decompose_time_series(series, frequency):
    """
    Decompose a time series and plot it in the console
    Arguments: 
        series: series. Time series that we want to decompose
    Outputs: 
        Decomposition plot in the console
    """
    result = seasonal_decompose(series, model='additive', freq = frequency)
    result.plot()
    plt.show()
    
def time_series_train_test_split(time_series, train_split_fraction):
    """
    Split the data into training and test set.
    """
    split_index = int(round(time_series.shape[0]*train_split_fraction, 0))
    train_set = time_series[:split_index]
    test_set = time_series[:-split_index]
    return train_set, test_set

def sarima_parameter_search(search_range, seasonal = [12]):
    """
    Get all of the parameter combinations for a SARIMA model.
    """
    p = q = d = range(0, search_range)
    trend = ['n','c','t','ct']
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq_combinations = [(x[0], x[1], x[2], x[3], x[4]) for x in list(itertools.product(p, d, q, seasonal, trend))]
    return pdq, seasonal_pdq_combinations

def seasonal_arima_model(time_series, order, seasonal_order, trend):
    """
    Generate a seasonal ARIMA model using a set of hyperparameters. Returns the model fit, and the 
    associated model AIC and BIC values.
    """ 
    try:
        model = sm_api.tsa.SARIMAX(time_series, 
                                   order=order, 
                                   seasonal_order=seasonal_order, 
                                   trend = trend,
                                   enforce_stationarity=False, 
                                   enforce_invertibility=False)
        model_fit = model.fit()
        #Print the model results
        print(model_fit.summary())
        return model_fit, model_fit.aic, model_fit.bic
    except:
        print("Could not fit with the designated model parameters")
        return None, None, None
    
def fit_predictions(model_fit, steps_out_to_predict, actual_values):
    """
    This function predicts the SARIMA model out a certain designated number of steps,
    and compares the predictions to the actual values. The root mean squared error and
    the mean absolute error are calculated, comparing the predicted and actual values.
    The function returns the predicted values and their respective confidence intervals.
    Args:
        model_fit:  SARIMA model.
        steps_out_to_predict: Int. Number of steps out to predict the time series.
        actual_values: Series of actual time series values.
    Outputs:
        mean_predicted_values: Series of predicted time series values.
        confidence_interval_predicted_values: Dataframe, containing upper and lower thresholds of the
        confidence interval
    """
    predicted_values = model_fit.get_forecast(steps=steps_out_to_predict)
    mean_predicted_values = predicted_values.predicted_mean
    confidence_interval_predicted_values = predicted_values.conf_int()
    #Compare the actual to the predicted values using RMSE and MAE metrics
    rmse, mae = quantify_rmse_mae(mean_predicted_values, actual_values)
    print("Root mean squared error: ", str(rmse))
    print("Mean absolute error: ", str(mae))
    return mean_predicted_values, confidence_interval_predicted_values
    
def quantify_rmse_mae(predicted_values, actual_values):
    """
    This function calculates the root mean squared error and mean absolute error for 
    the predicted values, when compared to the actual values. These helps help us to
    gauge model performance. 
    Args:
        predicted_values: Series of predicted time series values.
        actual_values: Corresponding series of actual time series values.
    Outputs:
        rmse: Float. Root mean squared error.
        mae: Float. Mean absolute error.
    """
    #calcuate the mean squared error of the model
    rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
    #Calculate the mean absolute error of the model 
    mae = mean_absolute_error(actual_values, predicted_values)
    #Return the MSE and MAE for the model
    return rmse, mae

def plot_results(mean_predicted_values, confidence_interval_predicted_values, time_series):
    """
    This function plots actual time series data against SARIMA model-predicted values. 
    We include the confidence interval for the predictions. 
    Args:
        mean_predicted_values: Series of float values. The model-predicted values.
        confidence_interval_predicted_values: Pandas dataframe, containing the lower and
        upper confidence intervals.
        time_series: Series of float values. Actual time series values that we want to graph
    Outputs:
        None. Plot of the time series values, as well as the predicted values and associated 
        confidence interval.
    """
    ax = time_series.plot(label='Observed')
    mean_predicted_values.plot(ax=ax, label = 'Forecast', alpha=.7, figsize=(14, 4))
    ax.fill_between(confidence_interval_predicted_values.index,
                    confidence_interval_predicted_values.iloc[:, 0],
                    confidence_interval_predicted_values.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date Index')
    ax.set_ylabel('Value')
    plt.legend()
    plt.show()
    
#if __name__== "__main__":
def main():
    """
    Run main script
    """
    #Create EIA API using your specific API key
    api_key = "YOUR API KEY HERE"
    api = eia.API(api_key)
    #Declare desired series ID
    series_ID='TOTAL.GEEGPUS.M'
    df = retrieve_time_series(api, series_ID)
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index':'Date',
                   df.columns[1]:'Geothermal_net_generation'}, inplace=True)
    #Convert the Date column into a date object
    df['Date'] = df['Date'].str.rstrip()
    df['Date'] = df['Date'].str.replace(' ', '-')
    df['Date']=pd.to_datetime(df['Date'], format='%Y-%m')
    #Plot the time series
    plot_data(df, 'Date', 
              'Geothermal_net_generation', 
              'Net Generation for Geothermal over Time')
    #Decompose the time series to determine seasonality/trend
    decompose_time_series(df['Geothermal_net_generation'], 12)
    
    ##### SARIMA MODEL #####
    #Run hyperparameter search on SARIMA model
    order_combos, seasonal_order_combos = sarima_parameter_search(search_range = 2)
    #Split the data into training and test sets (75/25 split)
    training_set, test_set = time_series_train_test_split(time_series = df['Geothermal_net_generation'], 
                                                          train_split_fraction = .75)
    lowest_aic_val = 100000000000
    #Generate  model for each of hyperparameter combination in a loop
    for order_combo in order_combos:
        for seasonal_order_combo in seasonal_order_combos:
            #Convert the combination to list format
            seasonal_order_combo = list(seasonal_order_combo)
            #Generate the SARIMA model
            model_fit, model_aic, model_bic = seasonal_arima_model(time_series = training_set, 
                                                                    order = order_combo, 
                                                                    seasonal_order = seasonal_order_combo[0:4],
                                                                    trend = seasonal_order_combo[-1])
            #Test model performance, and keep running tab of best performing model
            #Set with the newest value if the lowest_aic_value hasn't yet been calculated (on first run),
            #or if the newly calculated model AIC is lower than the lowest calculated AIC value
            if (model_aic < lowest_aic_val):
                lowest_aic_val = model_aic
                best_model = model_fit
                best_order = order_combo
                best_seasonal_order = seasonal_order_combo
    #Print the best model parameters after the 
    print("Best model paramaters: order-- ", best_order, ", seasonal order-- ", best_seasonal_order)  
    print(best_model.summary())
    #Run the data on the test set to gauge model performance
    mean_predicted_values, confidence_interval_predicted_values = fit_predictions(best_model, 
                                                                                  len(test_set), 
                                                                                  test_set)
    #Plot the predictions against the real data
    plot_results(mean_predicted_values, 
                 confidence_interval_predicted_values, 
                 df['Geothermal_net_generation'][400:])