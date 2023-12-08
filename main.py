import os
import pandas as pd
import sqlite3
import requests
import common
from sklearn.model_selection import train_test_split

# Téléchargez le fichier zip à partir de l'URL
url = 'https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip'
r = requests.get(url)

# Écrivez le contenu dans un fichier zip
path = 'data/New_York_City_Taxi_Trip_Duration.zip'
with open(path, 'wb') as f:
    f.write(r.content)

# Lisez les données à partir du fichier zip
data = pd.read_csv(path, compression='zip')

# Supprimez le fichier zip après avoir lu les données
os.remove(path)

# Supprimez les colonnes inutiles
data = data.drop(columns=['id', 'dropoff_datetime'])

# Convertissez la colonne 'pickup_datetime' en datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

# Divisez les données en ensembles de formation et de test
RANDOM_STATE = 42
data_train, data_test = train_test_split(data, test_size=0.3, random_state=RANDOM_STATE)

# Créez le répertoire de la base de données si nécessaire
db_dir = os.path.dirname(common.DB_PATH)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Enregistrez les données de formation et de test dans une base de données
print(f"Saving train and test data to a database: {common.DB_PATH}")
with sqlite3.connect(common.DB_PATH) as con:
    data_train.to_sql(name='train', con=con, if_exists="replace")
    data_test.to_sql(name='test', con=con, if_exists="replace")
