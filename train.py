import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import common

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X = data_train.drop(columns=['trip_duration'])
    y = common.transform_target(data_train['trip_duration'])
    return X, y

def fit_model(X, y):
    print(f"Fitting a model")
    X = common.step1_add_features(X)
    X = common.step2_add_features(X)
    num_features = ['log_distance_haversine', 'hour',
                    'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    params = {'regression__alpha': np.logspace(-2, 2, 20)}
    gs_ridge = GridSearchCV(pipeline, [params], scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1, verbose=2)
    gs_ridge.fit(X[train_features], y)

    best_alpha_ridge = gs_ridge.best_params_["regression__alpha"]
    print("Best alpha (Ridge) = %.4f" % best_alpha_ridge)

    y_pred = gs_ridge.predict(X[train_features])
    score = mean_squared_error(y, y_pred)
    print(f"Score on train data {score:.2f}")
    return gs_ridge





if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    X_train = common.preprocess_data(X_train)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)
