from datasets import load_dataset
from cli import CLI
import ast
from data import get_sent_list
import os
import json
import re

def evaluate_sent_order(experiment_file):
    exp_data = load_dataset("json", data_files=experiment_file, split="train")

    eval_data = exp_data.map(evaluate_order_answer, batched=True, remove_columns=exp_data.column_names)

    eval_file = experiment_file.replace("outputs", "eval")
    # TODO: fix repo list
    os.makedirs(f"{"/".join(eval_file.split("/")[:-1])}", exist_ok = True)
    eval_data.to_json(eval_file)

    acc_mean = float(eval_data.with_format("pandas")["accuracy"].mean())
    print("Mean accuracy:", acc_mean)
    with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
        json.dump({eval_file.split("/")[-1]: acc_mean}, f)


def evaluate_order_answer(batch):
    pred_lists = []
    acc = []
    gold_sents = []

    for i in range(len(batch["storyid"])):
        pred = extract_answer_list(batch["output"][i])
        pred_lists.append(pred)
        acc.append(compute_accuracy_order(pred, batch, i))
        gold_sents.append(get_sent_list(batch, i))
    
    return {"accuracy": acc, "predictions": pred_lists, "gold": gold_sents}

def extract_answer_list(model_out):
    try:
        model_answer = model_out.split("Answer:\n")[-1].strip()
        model_pred = ast.literal_eval(model_answer)
        return model_pred
    except:
        str_match = re.search(r"(\[.*?\])", model_answer) # only use first guess by the model

        try:
            return ast.literal_eval(str_match.group(1))
        except:
            return ["", "", "", ""]

def compute_accuracy_order(pred, batch, i):
    acc = 0.0

    for sent_order in range(len(pred)):
        if pred[sent_order] == batch[f"sentence{sent_order+1}"][i]:
            acc += 1
    
    return acc/len(pred) # 4 sentences

if __name__ == "__main__":
    cli = CLI()
    evaluate_sent_order(cli.experiment_file)