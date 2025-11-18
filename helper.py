import csv 
import json
from datasets import load_dataset, Dataset
import pandas as pd
import os

def check_SCT_ROC_overlap():
    val_sents = set()
    with open("data/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv", "r", encoding = "utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        for row in reader:
            val_sents.add(row[1])

    test_sents = set()
    with open("data/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv", "r", encoding = "utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        for row in reader:
            test_sents.add(row[1])

    sct_sents = val_sents.union(test_sents)

    roc_sents = set()
    with open("data/ROCStories_winter2017 - ROCStories_winter2017.csv", "r", encoding = "utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        for row in reader:
            roc_sents.add(row[1])
    print("intersect:", len(test_sents.intersection(roc_sents)))
    print("intersect:", val_sents.intersection(roc_sents))

def debug_json(input_file):
    dataset = load_dataset(
        "json", 
        data_files=input_file, 
        split="train",
        streaming=True  # This prevents loading everything into memory
    )

    with open(input_file, 'r') as infile, open("test.json", 'w') as outfile:
        for i, line in enumerate(infile):
            try:
                data = json.loads(line)
                
                # Convert all values to strings to ensure consistency
                # Or apply specific conversions for known problematic fields
                cleaned_data = {}
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        # Convert complex objects to JSON strings
                        cleaned_data[key] = json.dumps(value)
                    else:
                        cleaned_data[key] = str(value)  # or keep as is
                
                outfile.write(json.dumps(cleaned_data) + '\n')
            except Exception as e:
                print(f"Error processing line {i}: {e}")


if __name__ == "__main__":
    #check_SCT_ROC_overlap()
    debug_json("full_test.jsonl")