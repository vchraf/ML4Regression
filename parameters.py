from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

EXP_NAME        = "ML4Regression"
TRACKING_URL    = "http://127.0.0.1:5000"
MODEL           = LinearRegression
DATASET         = "./data/data"
SEARCH_SPACE    = {'bootstrap': hp.choice('bootstrap', [True, False]),
                 'max_depth': scope.int(hp.quniform('max_depth', 10, 100, 10)),
                 'max_features': hp.choice('max_features', ['auto', 'sqrt']),
                 'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
                 'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
                 'n_estimators': scope.int(hp.quniform('n_estimators', 100, 3000, 250))}
