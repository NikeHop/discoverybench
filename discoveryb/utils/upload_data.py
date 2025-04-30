"""
Create datasets from the test directory and upload them to huggingface
"""

import glob
import json
import pandas as pd 
from datasets import Dataset, DatasetDict

# Train Dataset
"""
train_filepath = (
    "/home/niklas/Desktop/Work/PhD/Projects/discoverybench/discoverybench/real/train"
)

dataset = {
    "id": [],
    "domain": [],
    "workflow_tags": [],
    "domain_knowledge": [],
    "datasets": [],
    "gold_workflow": [],
    "question_type": [],
    "question": [],
    "gold_hypothesis": [],
}

for directory in glob.glob(train_filepath + "/*"):
    area = directory.split("/")[-1]
    metadata_files = glob.glob(directory + "/metadata_*.json")
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        meta_id = metadata_file.split("/")[-1].split(".")[0][-1]
        
        for query in metadata["queries"][0]:
            query_id = query["qid"]
            idd = f"{area}_{meta_id}_{query_id}"
            dataset["id"].append(idd)
            dataset["domain"].append(metadata["domain"])
            dataset["workflow_tags"].append(metadata["workflow_tags"])
            dataset["domain_knowledge"].append(metadata["domain_knowledge"])
            dataset["datasets"].append(metadata["datasets"])
            dataset["gold_workflow"].append(metadata["workflow"])
            dataset["question_type"].append(query["question_type"])
            dataset["question"].append(query["question"])
            dataset["gold_hypothesis"].append(query["true_hypothesis"])

train_dataset = Dataset.from_dict(dataset)
train_dataset.push_to_hub("nhop/discoverybench_alias")

"""
test_filepath = (
    "/home/niklas/Desktop/Work/PhD/Projects/discoverybench/discoverybench/real/test"
)

dataset = {
    "id": [],
    "domain": [],
    "workflow_tags": [],
    "domain_knowledge": [],
    "datasets": [],
    "hypotheses": [],
    "gold_workflow": [],
    "question_type": [],
    "question": [],
    "gold_hypothesis": [],
}

df = pd.read_csv("/home/niklas/Desktop/Work/PhD/Projects/discoverybench/eval/answer_key_real.csv")

for directory in glob.glob(test_filepath + "/*"):
    print(f"Directory: {directory}")
    area = directory.split("/")[-1]
    metadata_files = glob.glob(directory + "/metadata_*.json")
    print(f"Found {len(metadata_files)} metadata files")
    for metadata_file in metadata_files:
        print(f"Metadata file: {metadata_file}")    
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        meta_id = metadata_file.split("/")[-1].split(".")[0].split("_")[-1]
        row = df[df["metadataid"] == meta_id]
        
        print(f"Number of queries {len(metadata['queries'][0])}")
        #print(metadata["queries"])
        for query in metadata["queries"][0]:

            query_id = query["qid"]
            idd = f"{area}_{meta_id}_{query_id}"

           

            row = df[(df["dataset"] == area) & (df["metadataid"] == int(meta_id)) & (df["query_id"] == int(query_id))]
           
            hypotheses = row["gold_hypo"].values
            
            if len(hypotheses)!=1:
                print(f"Wrong number of hypotheses for {idd} {len(hypotheses)}")
                continue
                
            dataset["id"].append(idd)
            dataset["domain"].append(metadata["domain"])
            dataset["workflow_tags"].append(metadata["workflow_tags"])

            if "domain_knowledge" in metadata:
                dataset["domain_knowledge"].append(metadata["domain_knowledge"])
            else:
                dataset["domain_knowledge"].append("")

            dataset["datasets"].append(metadata["datasets"])
            dataset["gold_workflow"].append("")
            dataset["question_type"].append(query["question_type"])
            dataset["question"].append(query["question"])
            
            dataset["gold_hypothesis"].append(hypotheses[0])
            
test_dataset = Dataset.from_dict(dataset)
test_dataset.push_to_hub("nhop/discoverybench_alias")
