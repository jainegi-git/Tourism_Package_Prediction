import pandas as pd
import os
import joblib
import xgboost as xgb
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from huggingface_hub import hf_hub_download

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")

repo_id = "jai-negi8/Tourism-Package-Prediction"

# Download processed data from HF because GitHub Actions jobs don't share local files
for filename in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=".", token=HF_TOKEN)

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

numeric_features = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

class_weight = (pd.Series(ytrain).value_counts()[0] / pd.Series(ytrain).value_counts()[1])
preprocessor = make_column_transformer((StandardScaler(), numeric_features), (OneHotEncoder(handle_unknown='ignore'), categorical_features))
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {'xgbclassifier__n_estimators': [50, 100], 'xgbclassifier__max_depth': [3, 4], 'xgbclassifier__learning_rate': [0.05, 0.1]}
model_pipeline = make_pipeline(preprocessor, xgb_model)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment")

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    best_model = grid_search.best_estimator_
    
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/best_model.joblib")
    mlflow.sklearn.log_model(best_model, "best_model")
    print("Model trained and saved locally.")
