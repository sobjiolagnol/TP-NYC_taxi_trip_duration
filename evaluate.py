import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import common


def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['trip_duration'])
    y = common.transform_target(data_test['trip_duration'])
    return X, y

def evaluate_model1(model, X, y):
    print(f"---------Evaluating the model 1----------")
    X = common.step1_add_features(X)
    X = common.step2_add_features(X)
    num_features = ['log_distance_haversine', 'hour',
                    'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features
    y_pred = model.predict(X[train_features])
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    print(f"----Test RMSE: {rmse:.4f}")
    print(f"----Test R2: {r2:.4f}")

if __name__ == "__main__":
    X_test, y_test = load_test_data(common.DB_PATH)
    model = common.load_model(common.MODEL_PATH)
    evaluate_model1(model, X_test, y_test)

