import csv 
import json
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
import os
from collections import defaultdict
import random
from evaluate import evaluate_sent_order

def check_SCT_ROC_overlap():
    """
    This function checks whether the SCT instances are included in the ROCStories corpus.
    """
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

def check_left_insts(outdir: str):
    """
    This function checks how many instances are still part of the ongoing conversation in the multi-agent framework (= how many outputs do not contain "STOP" in the final answer).

    @params
        outdir: path to the output dataset
    """
    exp_data = load_from_disk(outdir)

    filtered_data = exp_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)

    print(len(filtered_data))

def inspect_instances(outdir: str, rand_idx: int = None):
    """
    This function prints the output and evaluation results for a random or specified instance.

    @params
        outdir: path to output dataset
        rand_idx: idx of the instance, random if None
    """
    # load output data
    exp_data = load_from_disk(outdir)
    
    # load evaluation data
    eval_file = outdir.replace("outputs", "eval") + ".jsonl"
    eval_data = load_dataset("json", data_files=eval_file, split="train")
    
    # assign random idx
    if rand_idx == None:
        rand_idx = random.randint(0, 100-1)

    # select instances
    out = exp_data.select([rand_idx])[0]
    eval = eval_data.select([rand_idx])[0]

    # print information 
    print("Output File: ", outdir)
    print("IDX: ", rand_idx)
    for k, v in out.items():
        if k == "turns":
            print(f"\n{k}:\n")
            for turn in v:
                print("\n")
                print(turn)
        else:
            print(f"\n{k}:\n{v}")
    for k,v in eval.items():
        print(f"\n{k}:\n{v}")
    # print(out)
    # print(eval)

if __name__ == "__main__":
    #check_SCT_ROC_overlap()
    #check_left_insts("outputs/multi/Llama-3.2-1B-Instruct/dev/ckps/outputs_multi_Llama-3.2-1B-Instruct_dev_2025-11-18_11:43:21.920146_1")

    #qwen14_out = "outputs/single/Qwen3-8B/dev/outputs_single_Qwen3-8B_dev_2025-11-25_14:10:17.428416"
    qwen8_out = "outputs/single/Qwen3-32B/dev/outputs_93" # instances start with 0

    # gemini_out = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:22:00.500021"
    #llama_out = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743"
    # # evaluate_sent_order(outdir, "single")
    #inspect_instances(qwen14_out, 371)
    inspect_instances(qwen8_out, 8)
    #inspect_instances(llama_out, 4)

    # outdir = "outputs/multi/Qwen3-14B/dev/outputs_86"
    # evaluate_sent_order(outdir, "multi")