import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)
repo_id = "jai-negi8/Tourism-Package-Prediction"

# Load Data
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded from HF.")
except:
    print("HF load failed, using local file.")
    df = pd.read_csv("tourism_project/data/tourism.csv")

# Basic Cleaning
df.drop(columns=['CustomerID'], errors='ignore', inplace=True)
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Encoding
le = LabelEncoder()
cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Split
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save and Upload
files = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}
for name, data in files.items():
    data.to_csv(name, index=False)
    api.upload_file(path_or_fileobj=name, path_in_repo=name, repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded {name}")
