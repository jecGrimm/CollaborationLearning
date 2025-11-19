import csv 
import json
from datasets import load_dataset, Dataset, load_from_disk
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

def check_left_insts(outdir):
    exp_data = load_from_disk(outdir)

    filtered_data = exp_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)

    print(len(filtered_data))

if __name__ == "__main__":
    #check_SCT_ROC_overlap()
    check_left_insts("outputs/multi/Llama-3.2-1B-Instruct/dev/ckps/outputs_multi_Llama-3.2-1B-Instruct_dev_2025-11-18_11:43:21.920146_1")