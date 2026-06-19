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
from huggingface_hub import HfApi

# Configuration
DATASET_URL = "https://huggingface.co/datasets/jai-negi8/Tourism-Package-Prediction/resolve/main/"
Xtrain = pd.read_csv(DATASET_URL + "Xtrain.csv")
ytrain = pd.read_csv(DATASET_URL + "ytrain.csv").values.ravel()

numeric_features = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

preprocessor = make_column_transformer((StandardScaler(), numeric_features), (OneHotEncoder(handle_unknown='ignore'), categorical_features))
xgb_model = xgb.XGBClassifier(random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {'xgbclassifier__n_estimators': [50, 100], 'xgbclassifier__max_depth': [3, 5]}

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "best_model.joblib")
    print("Model trained and saved as best_model.joblib")
