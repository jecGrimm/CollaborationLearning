import csv 
import json
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
import os
from collections import defaultdict
from evaluate import evaluate_sent_order
import random

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

def inspect_instances(outdir):
    exp_data = load_from_disk(outdir)
    
    eval_file = outdir.replace("outputs", "eval") + ".jsonl"
    eval_data = load_dataset("json", data_files=eval_file, split="train")

    rand_idx = random.randint(0, len(exp_data)-1)

    out = exp_data.select([rand_idx])[0]
    eval = eval_data.select([rand_idx])[0]

    print("Output File: ", outdir)
    print("IDX: ", rand_idx)
    for k, v in out.items():
        print(f"\n{k}:\n{v}")
    for k,v in eval.items():
        print(f"\n{k}:\n{v}")
    # print(out)
    # print(eval)

def evaluate_all_outputs():
    splitdir = "outputs/single/Llama-3.2-1B-Instruct/dev/"

    outdirs = os.listdir(splitdir)

    #evals = defaultdict()
    for outdir in outdirs:
        if outdir != "ckps" and not outdir.endswith("jsonl"):
            #evals[outdir] = evaluate_sent_order(splitdir + outdir, "single")
            print(f"\n{outdir}: {evaluate_sent_order(splitdir + outdir, "single")}")

    #print(evals)

if __name__ == "__main__":
    #check_SCT_ROC_overlap()
    #check_left_insts("outputs/multi/Llama-3.2-1B-Instruct/dev/ckps/outputs_multi_Llama-3.2-1B-Instruct_dev_2025-11-18_11:43:21.920146_1")

    #evaluate_all_outputs()

    outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743"
    # evaluate_sent_order(outdir, "single")
    inspect_instances(outdir)