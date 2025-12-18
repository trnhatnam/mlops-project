#!/usr/bin/env python
# coding: utf-8

import socket
import xgboost as xgb
from pathlib import Path
import mlflow
import pandas as pd
import pickle
import scipy.io
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import logging
from airflow.sdk import dag, task
import pendulum


DATA_FOLDER = Path("/opt/airflow/data")


# Get the specific logger for Airflow tasks
logger = logging.getLogger("airflow.task")

def read_dataframe(year,month):
    logger.info(f"Reading data for {year}-{month:02d}")
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    # Créer le dossier s'il n'existe pas
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    df_path = DATA_FOLDER / f'green_tripdata_{year}-{month:02d}.parquet'
    df.to_parquet(df_path, index=False)
    return str(df_path)

@task(multiple_outputs=True)
def load_data():
    logger.info(f"Loading dataframes...")
    year=2021
    month=pendulum.now().month
    q,mod = divmod(month+1, 12) if month!=11 else (0, month+1)

    df_train_path = read_dataframe(year=year, month=month)
    df_val_path = read_dataframe(year=year+q, month=mod)
    
    return {"train_path": df_train_path, "val_path": df_val_path}

@task(multiple_outputs=True)
def create_Xy(df_path, tag="train", dv_path=None):
    logger.info("Creating feature matrix")
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = 'duration'

    df = pd.read_parquet(df_path)
    dicts = df[categorical + numerical].to_dict(orient='records')
    y = df[target].values

    # 1. Gestion du DictVectorizer (dv)
    if dv_path is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
        dv_path = DATA_FOLDER / 'preprocessor.b'
        with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
    else:
        with open(dv_path, "rb") as f_in:
            dv = pickle.load(f_in)
        X = dv.transform(dicts)

    # 2. Sauvegarde de X (Matrice sparse Scipy)
    x_path = DATA_FOLDER / f'X_{tag}.npz'
    scipy.sparse.save_npz(x_path, X)

    # 3. Sauvegarde de y (Vecteur Numpy)
    y_path = DATA_FOLDER / f'y_{tag}.npy'
    np.save(y_path, y)

    # 4. On ne retourne que des STRINGS (chemins)
    return {
        "X_path": str(x_path), 
        "y_path": str(y_path), 
        "dv_path": str(dv_path)
    }



@task()
def train_model(X_train_path, y_train_path, X_val_path, y_val_path, dv_path):
    logger.info("Training model")

    # On transforme 'mlflow-server' en IP (ex: 172.18.0.5)
    mlflow_ip = socket.gethostbyname('mlflow-server')
    mlflow_url = f"http://{mlflow_ip}:5000"

    logger.info(f"MLflow tracking URI: {mlflow_url}")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run():
        X_train = scipy.sparse.load_npz(X_train_path)
        y_train = np.load(y_train_path)
        X_val = scipy.sparse.load_npz(X_val_path)
        y_val = np.load(y_val_path)

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact(dv_path)
        mlflow.xgboost.log_model(booster, name="models_mlflow")

        # return run_id from mlflow
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Model saved in run {run_id}")

@dag(
    schedule="0 0 1 * *",
    start_date=pendulum.datetime(2025, 12, 18, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def main():
    df_paths = load_data()
    features_train = create_Xy(df_paths["train_path"], tag="train")
    features_val = create_Xy(df_paths["val_path"], tag="val", dv_path=features_train["dv_path"])
    train_model(features_train["X_path"], features_train["y_path"], features_val["X_path"], features_val["y_path"], features_train["dv_path"])


dag_instance = main()
