from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from google.colab import userdata

HF_TOKEN = userdata.get("HF_TOKEN") # Get token securely from Colab secrets
api = HfApi(token=HF_TOKEN)

repo_id = "jai-negi8/Tourism-Package-Prediction"
repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload files in HF
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)
