import mlflow
import parameters
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

mlflow.set_tracking_uri(parameters.TRACKING_URL)
mlflow.set_experiment(parameters.EXP_NAME)
model = mlflow.pyfunc.load_model(f"models:/{parameters.MODEL_REGISTER}/Production")


app = Flask('duration-prediction')



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    _request = request.get_json()

    result = {
        'prix': model.predict(list(_request.values()))
    }
    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)