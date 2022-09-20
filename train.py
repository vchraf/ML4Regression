import ast
import parameters
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
def load_data(src):
    data = pd.read_parquet(src)
    train, test = train_test_split(data)

    X_train = train.drop(["price"], axis=1)
    X_test = test.drop(["price"], axis=1)
    y_train = train[["price"]]
    y_test = test[["price"]]
    return X_train, X_test, y_train, y_test


mlflow.set_tracking_uri(parameters.TRACKING_URL)
mlflow.set_experiment(parameters.EXP_NAME)


# client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

# EXP_ID=dict(mlflow.get_experiment_by_name(parameters.EXP_NAME))['experiment_id']
# run = client.search_runs(experiment_ids=EXP_ID, run_view_type=ViewType.ACTIVE_ONLY, max_results=1, order_by=["metrics.r2  DESC"])[0]
# model_params = run.data.params
# for key in model_params.keys():
#     try:
#         model_params[key] = ast.literal_eval(model_params[key])
#     except : pass



X_train, X_test, y_train, y_test = load_data(src=parameters.DATASET)

with mlflow.start_run():
    model = parameters.MODEL()
    print(model.__class__.__name__)
    mlflow.set_tag("modele", model.__class__.__name__)
    mlflow.set_tag("status", "final")
    # mlflow.log_params(best_params)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, y_pred)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "model", registered_model_name=model.__class__.__name__)

