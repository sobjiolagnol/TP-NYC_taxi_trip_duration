import pandas as pd
from sklearn.model_selection import train_test_split
import common
import os
import sqlite3

data = pd.read_csv('NYC_Taxi_Trip_Duration.csv')

data = data.drop(columns=['id'])
data = data.drop(columns=['dropoff_datetime'])
data['pickup_datetime'].dtype
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
RANDOM_STATE=42
data_train,data_test= train_test_split(data, test_size=0.3, random_state=RANDOM_STATE)


db_dir = os.path.dirname(common.DB_PATH)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

print(f"Saving train and test data to a database: {common.DB_PATH}")
with sqlite3.connect(common.DB_PATH) as con:
    # cur = con.cursor()
    # cur.execute("DROP TABLE IF EXISTS train")
    # cur.execute("DROP TABLE IF EXISTS test")
    data_train.to_sql(name='train', con=con, if_exists="replace")
    data_test.to_sql(name='test', con=con, if_exists="replace")