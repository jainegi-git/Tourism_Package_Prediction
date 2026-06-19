# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os
import datetime

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)
DATASET_PATH = "hf://datasets/jai-negi8/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)


#------------------------------------------------
#df = pd.read_csv("https://huggingface.co/datasets/jai-negi8/Tourism-Package-Prediction/raw/main/tourism.csv")
#------------------------------------------------

#check if dataset loaded correctly from huggingface
if df is not None:
    print("Dataset loaded successfully from Huggingface.")
else:
  # if data not loaded then load from local file
    print("HF data not loaded, checking local folder.")
    df = pd.read_csv("/content/tourism_project/data/tourism.csv")
    if df is not None:
      print("Dataset loaded from local folder successfully.")
    else:
      print("Data load was unsuccessful.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Encoding all the categorical from dataset
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])


# Define the target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert to CSV
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="jai-negi8/Tourism-Package-Prediction",
        repo_type="dataset"
    )
