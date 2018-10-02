import json
import pandas
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

"""
    This method loads the data for a specific station id, processes it and the train and evaluate
    a Random Forest Regressor algorithm to predict 3 different horizon. The function returns a  
"""
def evaluate(station_id, plot_gini_coef=False, use_time_features=False, show_plot=False):

    def index_to_hour(x):
        #TODO CP2 - return the hour information from the index x
        pass

    def index_to_weekday(x):
        # TODO CP2 - return the weekday information from the index x
        pass

    ## Load data
    #TODO CP1 load the data

    ## Process data (Note: data could also be normalized and/or one-hot encoded for better results)
    if use_time_features:
        #TODO CP2 extract time features (hour and dayofweek) and add it to the dataframe
        pass

    ## Split the data into train and test sets (important keep the temporal order)
    #TODO CP1 split the dataframe into train and test sets (75% - 25%)

    ## Split the train_set into X_train (input data), y_train (output data)
    #TODO CP1 split the train set to isolate input from output data

    ## Split the test_set into X_test (input data), y_test (output data, aka ground truth)
    # TODO CP1 split the test set to isolate input from output data

    ## Initialize the ML model
    #TODO CP1 initialize your RandomForestRegressor

    ## Train the model using the X_test and y_test data
    #TODO CP1 Train your model (model.fit() function)

    ## Perform the predictions for the X_test set
    #TODO CP1 Perform (and get) the predictions with the trained model

    ## Convert the received predictions (list of lists) to a dataframe (specify columns names)
    #TODO CP1 Convert the predictions back to a pandadataframe (you may use other solutions here)

    ## Compare the obtained predictions and the ground_truth(y_test_set)
    #TODO CP1 Compare the predictions and the groundtruth and measure the Mean Aboslute Error (of each horizon)
    mean_absolute_error = [0, 0, 0] #TODO CP1 - You can directly compute the mean on the error dataframe

    print("Mean absolute error: {}".format(mean_absolute_error))

    ## Plot Mean Absolute Error
    if show_plot:
        #TODO CP1 Generate the bar plot the plot for the mean absolute error per horizon
        plt.title('Mean absolute error (Station <{}>)'.format(station_id))
        plt.tight_layout()
        plt.show()

    ##Plot Gini coefficients and their importance
    if show_plot:
        # TODO CP1 Generate the bar plot for the feature importance from the model (Gini coefficients)
        plt.tight_layout()
        plt.title('Gini coefficients (Station <{}>)'.format(station_id))
        plt.show()

    return mean_absolute_error


""" 
    Load the data from a file and return it as a panda dataframe 
    Columns:  'bikes(t-20)', 'bikes(t-15)', 'bikes(t-10)', 'bikes(t-5)',  ## (int) number of bikes present at the station t minutes ago
       'holiday_type(t)',                                                 ## (int) the type of holiday (none, school, public)  
       'client_arrivals(t)', 'client_departures(t)',                      ## (int) the number of departures/arrivals at this station during the last 5 minutes
       'cur_temperature(t)', 'cur_humidity(t)', 'cur_cloudiness(t)',      ## (float) the current weather information  
       'cur_wind(t)', 
       'f3h_temperature(t)', 'f3h_humidity(t)', 'f3h_cloudiness(t)',      ## (float) the 3h forecasted weather information
       'f3h_wind(t)', 
       'client_departures_s43(t-20)', 'client_departures_s43(t-15)',      ## (int) the number of departures at the neighbouring stations (s_##) t minutes ago
       'client_departures_s43(t-10)',
       'client_departures_s43(t-5)', 'client_departures_s43(t)',
       'client_departures_s12(t-20)', 'client_departures_s12(t-15)',
       'client_departures_s12(t-10)', 'client_departures_s12(t-5)',
       'client_departures_s12(t)', 'client_departures_s22(t-20)',
       'client_departures_s22(t-15)', 'client_departures_s22(t-10)',
       'client_departures_s22(t-5)', 'client_departures_s22(t)',
       'client_departures_s15(t-20)', 'client_departures_s15(t-15)',
       'client_departures_s15(t-10)', 'client_departures_s15(t-5)',
       'client_departures_s15(t)', 'client_departures_s36(t-20)',
       'client_departures_s36(t-15)', 'client_departures_s36(t-10)',
       'client_departures_s36(t-5)', 'client_departures_s36(t)', 
       'bikes(t)',                                                      ## The current number of bikes at the station
       'bikes(t+15)', 'bikes(t+30)', 'bikes(t+60)'                      ## The future number of bikes at the station (those are the values to predict)
    
"""
def load_data(station_id):
    with open("data/station_data_{}.json".format(station_id)) as ifile:
        json_data = json.load(ifile)
        dataframe = pandas.read_json(json_data)
        # print("Columns of the dataframe: \n{}".format(dataframe.columns))
    return dataframe

"""
    When you run/debug the script from pycharm (or if you run it from your terminal), it starts here !
"""
if __name__ == "__main__":
    ##Execute the evaluate method for station 0 without extracting/using time features
    mae = evaluate(station_id=0, show_plot=True, use_time_features=False)

    #TODO CP2 - Use loops to evaluate all possible stations with and without using time features
