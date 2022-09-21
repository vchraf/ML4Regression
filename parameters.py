from hyperopt import hp
from hyperopt.pyll import scope
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

EXP_NAME        = "ML4Regression"
TRACKING_URL    = "http://127.0.0.1:5000"
MODEL           = CatBoostRegressor
MODEL_NAME      = MODEL().__class__.__name__
MODEL_REGISTER  = "RegressionModel"
DATASET         = "./data/data"
SEARCH_SPACE    = {'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
                 'depth': scope.int(hp.quniform('depth', 8, 15, 2)),
                 'l2_leaf_reg': hp.choice('l2_leaf_reg', [0, 1, 2, 3]),
                 'iterations': scope.int(hp.quniform('iterations', 100, 200, 10)),
                 'verbose':False}

