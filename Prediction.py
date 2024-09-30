import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox



def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def adfuller_test(y):
    adf_result = adfuller(y)

    print("ADF Statistic:", adf_result[0])
    print("P-Value:", adf_result[1])

def predict(prediction_date, prediction_hour, df):

    df['hour'] = df["dt_hour"].dt.hour
    df['date'] = df["dt_hour"].dt.date

    train_data = df[(df["dt_hour"].dt.date < prediction_date) | ((df["dt_hour"].dt.date == prediction_date) & (df["dt_hour"].dt.hour < prediction_hour))]
    test_data = df[(df["dt_hour"].dt.date == prediction_date) & (df["dt_hour"].dt.hour == prediction_hour)]

    tuple1 = (1, 1, 1)
    tuple2 = (1, 1, 1, 24)

    predictions = {}

    for poi in train_data['poi_id'].unique():

        train_subset = train_data[train_data['poi_id'] == poi].reset_index()
        test_subset = test_data[test_data['poi_id'] == poi].reset_index()
        train_len = len(train_subset.index)
        test_len = len(test_subset.index)

        if len(train_subset) > 150:
            try:

                model = SARIMAX(train_subset['count'], order=tuple1, seasonal_order=tuple2)
                sarima_result = model.fit(disp=False)

                forecast = sarima_result.get_prediction(train_len, train_len + test_len - 1).predicted_mean
                predictions[poi] = int(forecast.values)

                '''
                plt.figure(figsize=(12, 6))
                plt.plot(train_subset.index, train_subset['count'], label='Train Data')
                if not test_subset.empty:
                    plt.plot(test_subset.index, test_subset['count'], label='Test Data', color='gray')
                plt.title(f'POI {poi} - Actual vs Predicted Counts on {prediction_date} {prediction_hour}:00')
                plt.legend()
                plt.show()'''

            except Exception as e:
                print(f"Could not fit SARIMA model for POI {poi}: {e}")
        else:
            print(f"Not enough data to train SARIMA model for POI {poi}")
    print(list(predictions.items()))
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['poi_id', 'Predicted_Count'])
    predictions_df["poi_id"] = predictions_df["poi_id"].astype(object)
    df = df[(df['dt_hour'].dt.date == prediction_date)&(df['dt_hour'].dt.hour == prediction_hour)]
    df = pd.merge(df,predictions_df, on="poi_id", how="left")
    df.to_csv("Join file.csv")
    df['max_demand'] = df.groupby(['dt_hour', '2nd_cluster'])['Predicted_Count'].transform(max)
    df_Depot = df[df['Predicted_Count']==df['max_demand']]
    return df_Depot


if __name__ == '__main__':
    file_name = "Pickup Location Demand by time.csv"
    org_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Python code"
    f1 = os.path.join(org_path, file_name)

    data = pd.read_csv(f1)
    df = data
    df['dt_hour'] = pd.to_datetime(df['dt_hour'])
    df = df.sort_values(by=['dt_hour'], ignore_index=True)
    prediction_date = datetime(2022, 10, 24).date()
    prediction_hour = 5
    df_pred = predict(prediction_date, prediction_hour, df)





