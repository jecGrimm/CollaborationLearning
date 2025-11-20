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
from collections import defaultdict
import csv
from tqdm import tqdm

def evaluate_sent_order(experiment_file: str, framework: str, logger: ExperimentLog = ExperimentLog()):
    """
    This function evaluates the model predictions. 

    @params
        experiment_file: Huggingface folder containing the dataset with the outputs
        framework: single/multi-agent setup
        logger: experiment logging object 
    @returns
        acc_mean: mean accuracy over all instances
    """
    # load data
    #exp_data = load_dataset("json", data_files=experiment_file, split="train")
    #exp_data = load_from_disk(experiment_file).select([2856])
    exp_data = load_from_disk(experiment_file)
    
    # evaluate
    if framework == "single":
        eval_data = exp_data.map(evaluate_order_answer_single, batched=True, remove_columns=exp_data.column_names)
    else:
        eval_data = exp_data.map(evaluate_order_answer_multi, batched=True, remove_columns=exp_data.column_names)
    
    # save evaluation output
    if os.path.isdir(experiment_file):
        experiment_file = experiment_file + ".jsonl"

    eval_file = experiment_file.replace("outputs", "eval")

    os.makedirs(f"{"/".join(eval_file.split("/")[:-1])}", exist_ok = True)
    eval_data.to_json(eval_file)

    # report mean accuracy
    acc_mean = float(eval_data.with_format("pandas")["accuracy"].mean())

    logger.run_data.update({"mean_acc": acc_mean})
    print("Mean accuracy:", acc_mean)
    # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
    #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
    return acc_mean


def evaluate_order_answer_single(batch):
    """
    This function collect the evaluation results for the single-agent outputs.

    @param
        batch: batch of instances
    @returns
        dictionary with the storyids, accuracy per instance, model prediction, gold standard
    """
    pred_lists = []
    acc = []
    gold_sents = []

    for i in range(len(batch["storyid"])):
        pred = extract_answer_list(batch["output"][i])
        pred_lists.append(pred)
        acc.append(compute_accuracy_order(pred, batch, i))
        gold_sents.append(get_sent_list(batch, i))
    
    return {"storyid": batch["storyid"], "accuracy": acc, "predictions": pred_lists, "gold": gold_sents}

def evaluate_order_answer_multi(batch):
    """
    This function collect the evaluation results for the multi-agent outputs.

    @param
        batch: batch of instances
    @returns
        dictionary with the storyids, accuracy per instance, model prediction, gold standard
    """
    pred_lists = []
    acc = []
    gold_sents = []

    for i in range(len(batch["storyid"])):
        pred = extract_answer_list(batch["turns"][i][-1])
        pred_lists.append(pred)
        acc.append(compute_accuracy_order(pred, batch, i))
        gold_sents.append(get_sent_list(batch, i))
    
    return {"storyid": batch["storyid"], "accuracy": acc, "predictions": pred_lists, "gold": gold_sents}

def extract_answer_list(model_out: str):
    """
    This function extract the list with predictions from the model output string. 

    @param
        model_out: output of the model
    @returns
        list with the model prediction
    """
    try:
        model_answer = clean_out(model_out)
        # model only returned output list
        model_pred = ast.literal_eval(model_answer)
        
        return clean_preds(model_pred)
    except:
        # extract list from model output
        str_match = re.search(r"(\[.*?\])", model_answer) # only use first guess by the model

        if str_match:
            str_list = clean_match(str_match.group(1))
        try:
            return clean_preds(ast.literal_eval(str_list))
        except:
            return ["", "", "", "", ""]
        
def clean_out(model_out: str):
    """
    This function prepares the model output for the extraction of the prediction list.

    @params
        model_out: full output of the model
    @returns
        model_answer: model answer as single-line string
    """
    model_answer = model_out.split("Answer:\n")[-1].strip()
    # remove newlines to make matching possible
    model_answer = re.sub("\n", "", model_answer)
    return model_answer

def clean_match(str_list: str):
    """
    This function prepares the extracted string of the prediction list for the transformation into a python list.

    @params
        str_list: string match of the prediction list
    @returns
        str_list: string match with escaped quotation marks and as raw string
    """
    # escape single quote strings
    str_list = re.sub(r"\b'\b", r"\\'", str_list)
    str_list = fr"{str_list}"
    return str_list

def clean_preds(model_pred: list):
    """
    This function extracts the predictions from the output list.

    @params
        model_pred: extracted list from the model output
    @returns 
        cleaned_pred: list with model predictions in the correct format
    """
    # get types of items in the list
    pred_types = [(i, type(item)) for i, item in enumerate(model_pred) if type(item) != str]
    cleaned_pred = []

    if len(pred_types) != 0:
        # Prioritize tuples
        type_i = 0
        max_i = len(pred_types)
        while type_i < max_i:
            # type of the current item
            pred_type = pred_types[type_i][1]
            # index of the item in the model prediction
            model_i = pred_types[type_i][0]
            if pred_type in [tuple, list]:
                if len(model_pred[model_i]) >= 5:
                    # only use the first list in the prediction 
                    cleaned_pred = list(model_pred[model_i]) 
                    type_i = max_i
                else:
                    # extract str 
                    if len(model_pred) == 5:
                        # extract strings from enumerated lists
                        if type(model_pred[model_i][0]) == str and type(model_pred[model_i][1]) == int:
                            cleaned_pred = [item[0] for item in model_pred]
                            type_i = max_i
                        elif type(model_pred[model_i][1]) == str and type(model_pred[model_i][0]) == int:
                            cleaned_pred = [item[1] for item in model_pred]
                            type_i = max_i

                    # extract first list from multiple string lists
                    if type(model_pred[model_i][0]) == str and type(model_pred[model_i][1]) == str:
                        cleaned_pred = list(model_pred[0])
                        type_i = max_i
            elif pred_type == int:
                cleaned_pred = [""] * 5 # do not extract numbers due to wrong instruction following
            type_i += 1
    else:
        # turn model prediction into list if the format is already correct
        cleaned_pred = list(model_pred)

    return cleaned_pred[:5] # only use first 5 sentences

def compute_accuracy_order(pred: list, batch, i: int):
    """
    This function compares the prediction with the gold standard.

    @params
        pred: model prediction
        batch: batch of instances
        i: index of the prediction instance
    @returns
        accuracy of the model prediction (percentage of correctly positioned sentences)
    """
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
    
def compare_instance_acc(eval_files: dict):
    """
    This function compares a bunch of experiments against each other.

    @params
        eval_files: dictionary mapping the experiment name to the evaluation file.
    """
    # load datasets
    eval_datasets = defaultdict()
    results = defaultdict(lambda: defaultdict(float))
    for exp_name, eval_file in eval_files.items():
        #eval_datasets[exp_name] = load_dataset("json", data_files=eval_file, split="train").select(range(10))
        eval_data = load_dataset("json", data_files=eval_file, split="train")

        # store mean accuracy per experiment
        results[exp_name]["mean_acc"] = float(eval_data.with_format("pandas")["accuracy"].mean())
        eval_datasets[exp_name] = eval_data

    # compare evaluation on instance level
    for i in tqdm(range(len(eval_data)), desc="Processing instances", total = len(eval_data)):
        all_acc = defaultdict()

        # check if accuracy is the same for all experiments
        same = True
        comp_acc = eval_data["accuracy"][i]

        for curr_exp, curr_eval_data in eval_datasets.items():
            curr_acc = curr_eval_data["accuracy"][i]
            all_acc[curr_exp] = curr_acc

            if curr_acc != comp_acc:
                same = False # accuracy between experiments differs

            if curr_eval_data["predictions"][i] == ["", "", "", "", ""]:
                # check whether model returned prediction in the correct format
                results[curr_exp]["wrong_format"] += 1

        # compare accuracies between experiments
        for curr_exp in eval_datasets.keys():
            if max(all_acc.values()) == all_acc[curr_exp]:
                results[curr_exp]["max_acc"] += 1 # experiment has the best accuracy

            if min(all_acc.values()) == all_acc[curr_exp]:
                results[curr_exp]["min_acc"] += 1 # experiment has the worst accuracy

            if same:
                results[curr_exp]["equal_acc"] += 1 # instance is not affected by experiment

    # save comparison data in csv
    with open(f"{"/".join(eval_files[exp_name].split("/")[:-1])}/eval_compare.csv", 'w', encoding="utf-8") as csvfile:
        fieldnames = ["experiment_name", "candidates", "experiment_file", "mean_acc", "max_acc", "equal_acc", "min_acc", "wrong_format"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        
        writer.writeheader()
        for exp, file in eval_files.items():
            writer.writerow({
                "experiment_name": exp, 
                "candidates": list(eval_datasets.keys()), 
                "experiment_file": file, 
                "mean_acc": results[exp]["mean_acc"], 
                "max_acc": results[exp]["max_acc"]/len(eval_data), 
                "equal_acc": results[exp]["equal_acc"]/len(eval_data), 
                "min_acc": results[exp]["min_acc"]/len(eval_data),
                "wrong_format": results[exp]["wrong_format"]/len(eval_data)
            })

def evaluate_all_outputs(splitdir = "outputs/single/Llama-3.2-1B-Instruct/dev/"):
    """
    This function evaluates all outputs in a directory.

    @params
        splitdir: directory with the output files
    @overwrites all current evaluation files
    """
    outdirs = os.listdir(splitdir)

    #evals = defaultdict()
    for outdir in outdirs:
        if outdir != "ckps" and not outdir.endswith("jsonl"):
            #evals[outdir] = evaluate_sent_order(splitdir + outdir, "single")
            print(f"\n{outdir}: {evaluate_sent_order(splitdir + outdir, "single")}")

    #print(evals)

if __name__ == "__main__":
    cli = CLI()
    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-14_10:24:56.857093.jsonl"
    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_08:48:25.663742"

    # outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881"
    # #evaluate_sent_order("test.jsonl", "single")
    # evaluate_sent_order(outfile, "single")

    #outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_13:10:53.603867"
    #outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743"
    #evaluate_sent_order(outdir, "single")

    prompt_dev_files = {
        "self": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881.jsonl",
        "gemini": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:22:00.500021.jsonl",
        "llama": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743.jsonl"
    }
    compare_instance_acc(prompt_dev_files)