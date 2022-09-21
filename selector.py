import parameters

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


mlflow.set_tracking_uri(parameters.TRACKING_URL)
mlflow.set_experiment(parameters.EXP_NAME)
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
models = {}
for version in client.search_model_versions(f"name='{parameters.MODEL_REGISTER}'"):
    models[version.version] = mlflow.get_run(run_id=version.run_id).data.metrics['r2']
models = list(dict(sorted(models.items(), key=lambda item: item[1],reverse=True)).keys())


client.transition_model_version_stage(
    name=parameters.MODEL_REGISTER,
    version=models.pop(0),
    stage="Production",
    archive_existing_versions=True
)
for version in models:
    client.transition_model_version_stage(name=parameters.MODEL_REGISTER,version=version,stage="Archived")
