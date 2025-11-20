from datasets import load_dataset, Dataset, load_from_disk
from cli import CLI
import ast
from data import get_sent_list
import os
import json
import re
import pandas as pd
from log import ExperimentLog
import os

def evaluate_sent_order(experiment_file, framework, logger = ExperimentLog()):
    #exp_data = load_dataset("json", data_files=experiment_file, split="train")
    #exp_data = load_from_disk(experiment_file).select([2856])
    exp_data = load_from_disk(experiment_file)

    if framework == "single":
        eval_data = exp_data.map(evaluate_order_answer_single, batched=True, remove_columns=exp_data.column_names)
    else:
        eval_data = exp_data.map(evaluate_order_answer_multi, batched=True, remove_columns=exp_data.column_names)
    
    if os.path.isdir(experiment_file):
        experiment_file = experiment_file + ".jsonl"

    eval_file = experiment_file.replace("outputs", "eval")

    os.makedirs(f"{"/".join(eval_file.split("/")[:-1])}", exist_ok = True)
    eval_data.to_json(eval_file)

    acc_mean = float(eval_data.with_format("pandas")["accuracy"].mean())

    logger.run_data.update({"mean_acc": acc_mean})
    print("Mean accuracy:", acc_mean)
    # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
    #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
    return acc_mean


def evaluate_order_answer_single(batch):
    pred_lists = []
    acc = []
    gold_sents = []

    for i in range(len(batch["storyid"])):
        pred = extract_answer_list(batch["output"][i])
        pred_lists.append(pred)
        acc.append(compute_accuracy_order(pred, batch, i))
        gold_sents.append(get_sent_list(batch, i))
    
    return {"accuracy": acc, "predictions": pred_lists, "gold": gold_sents}

def evaluate_order_answer_multi(batch):
    pred_lists = []
    acc = []
    gold_sents = []

    for i in range(len(batch["storyid"])):
        pred = extract_answer_list(batch["turns"][i][-1])
        pred_lists.append(pred)
        acc.append(compute_accuracy_order(pred, batch, i))
        gold_sents.append(get_sent_list(batch, i))
    
    return {"accuracy": acc, "predictions": pred_lists, "gold": gold_sents}

def extract_answer_list(model_out):
    try:
        model_answer = clean_out(model_out)
        model_pred = ast.literal_eval(model_answer)
        
        return clean_preds(model_pred)
    except:
        str_match = re.search(r"(\[.*?\])", model_answer) # only use first guess by the model

        if str_match:
            str_list = clean_match(str_match.group(1))
        try:
            return clean_preds(ast.literal_eval(str_list))
        except:
            return ["", "", "", "", ""]
        
def clean_out(model_out):
    model_answer = model_out.split("Answer:\n")[-1].strip()
    # remove newlines to make matching possible
    model_answer = re.sub("\n", "", model_answer)
    return model_answer

def clean_match(str_list):
    # escape single quote strings
    str_list = re.sub(r"\b'\b", r"\\'", str_list)
    str_list = fr"{str_list}"
    return str_list

def clean_preds(model_pred):
    pred_types = [(i, type(item)) for i, item in enumerate(model_pred) if type(item) != str]
    cleaned_pred = []

    if len(pred_types) != 0:
        # Prioritize tuples
        type_i = 0
        max_i = len(pred_types)
        while type_i < max_i:
            pred_type = pred_types[type_i][1]
            model_i = pred_types[type_i][0]
            if pred_type in [tuple, list]:
                if len(model_pred[model_i]) >= 5:
                    # only use the first list item 
                    cleaned_pred = list(model_pred[model_i]) 
                    type_i = max_i
                else:
                    # extract str 
                    if len(model_pred) == 5:
                        if type(model_pred[model_i][0]) == str and type(model_pred[model_i][1]) == int:
                            cleaned_pred = [item[0] for item in model_pred]
                            type_i = max_i
                        elif type(model_pred[model_i][1]) == str and type(model_pred[model_i][0]) == int:
                            cleaned_pred = [item[1] for item in model_pred]
                            type_i = max_i
                    
                    if type(model_pred[model_i][0]) == str and type(model_pred[model_i][1]) == str:
                        cleaned_pred = list(model_pred[0])
                        type_i = max_i
            elif pred_type == int:
                cleaned_pred = [""] * 5 # do not extract numbers due to wrong instruction following
            type_i += 1
    else:
        cleaned_pred = list(model_pred)

    return cleaned_pred[:5] # only use first 5 sentences

def compute_accuracy_order(pred, batch, i):
    acc = 0.0

    if len(pred) > 0:
        for sent_order in range(len(pred)):
            sent_lbl = f"sentence{sent_order+1}"
            if sent_lbl in batch.keys():
                if pred[sent_order] == batch[f"sentence{sent_order+1}"][i]:
                    acc += 1
        
        return acc/5 # 5 sentences
    else:
        return 0.0

if __name__ == "__main__":
    cli = CLI()
    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-14_10:24:56.857093.jsonl"
    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_08:48:25.663742"

    # outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881"
    # #evaluate_sent_order("test.jsonl", "single")
    # evaluate_sent_order(outfile, "single")

    outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_13:10:53.603867"
    #outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743"
    evaluate_sent_order(outdir, "single")