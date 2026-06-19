from huggingface_hub import HfApi
import os
from google.colab import userdata # Import userdata

try:
    HF_TOKEN = userdata.get("HF_TOKEN") # Get token securely from Colab secrets
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN") # For github actions to work

api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/deployment",       # the local folder containing your files
    repo_id="jai-negi8/Tourism-Package-Prediction", # the target repo
    repo_type="space",                              # dataset, model, or space
    path_in_repo="",                                # optional: subfolder path inside the repo
)
