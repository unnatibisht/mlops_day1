import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# 1. Setup Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Advertising_Sales_Regression"
registered_model_name = "Sales_Prediction_Model"
mlflow.set_experiment(experiment_name)

# 2. Data Preparation
df = pd.read_csv(r"D:\mlops_day1\data\data.csv")
X, y = df[["TV", "radio", "newspaper"]], df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Candidate Models (LOG ONLY, DO NOT REGISTER YET)
models = {  
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForestRegressor": RandomForestRegressor(max_depth=5, random_state=42)
}

batch_runs = []

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        rmse = root_mean_squared_error(y_test, model.predict(X_test))

        mlflow.log_param("model_type", name)
        mlflow.log_metric("test_rmse", rmse)

        #Notice : NO registered_model_name here
        mlflow.sklearn.log_model(model, artifact_path="model")
        batch_runs.append((run.info.run_id, name, rmse))

# 4. Find the single best model from this batch
batch_runs.sort(key=lambda x: x[2])  # Sort by lowest RMSE
best_run_id, best_name, best_rmse = batch_runs[0]

# 5. Register ONLY the winning run as the challenger
client = MlflowClient()
challenger_model = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=registered_model_name
)
challenger_version = challenger_model.version

# Assign challenger alias
client.set_registered_model_alias(registered_model_name, "challenger", challenger_version)
print(f"Best batch run {best_run_id} registered as challenger version (v{challenger_version} with RMSE: {best_rmse:.4f})")

# 6. Challenger vs Champion Evaluation Gate
try:
    champion_info = client.get_model_version_by_alias(registered_model_name, "champion")
    champion_run = client.get_run(champion_info.run_id)
    champion_rmse = champion_run.data.metrics["test_rmse"]
    champion_version = champion_info.version

    print(f"Current Champion: Version {champion_version}  (RMSE: {champion_rmse:.4f}")

    if best_rmse < champion_rmse:
        client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
        print(f"Title Change! Challenger (v{challenger_version}) defeated (v{champion_version})")
    else:
        print(f"Defended! Champion (v{champion_version}) retains its title.")

except Exception:
    # first time running
    client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
    print(f"No existing champion found. Version {challenger_version}crowned as first Champion!")
