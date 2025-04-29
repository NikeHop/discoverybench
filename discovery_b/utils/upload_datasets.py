import glob 
from pathlib import Path
import pandas as pd 

from huggingface_hub import HfApi
import shutil 
import os 


# Collect all csv's of the training dataset 
test_filepath = Path("/home/niklas/Desktop/Work/PhD/Projects/discoverybench/discoverybench/real/test")

test_directories = glob.glob(str(test_filepath / "*"))

datasets = []
for directory in test_directories:
    files = glob.glob(directory + "/" + "*.csv")
    datasets += files 
    files = glob.glob(directory + "/" + "*.dta")
    datasets += files 

# Copy all the csv files into a folder in the current directory 
os.makedirs("real_test_datasets", exist_ok=True)

for file in datasets:
    shutil.copy(file, "real_test_datasets")

api = HfApi()

# Upload directory to huggingface 
api.upload_folder(
    repo_id="nhop/discoverybench_alias",
    folder_path="real_test_datasets",
    path_in_repo="real_test_datasets",
    repo_type="dataset"
)


