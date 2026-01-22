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
from scipy import stats

class Evaluator:
    def __init__(self, experiment_file: str, framework: str, logger: ExperimentLog = ExperimentLog(), idx = None, last = False):
        self.experiment_file = experiment_file
        if idx:
            self.exp_data = load_from_disk(experiment_file).select([idx])
        else:
            self.exp_data = load_from_disk(experiment_file)

        self.framework = framework
        
        self.logger = logger

        self.initiate_metric_lists()

        self.last = last
        
    def initiate_metric_lists(self):
        self.pred_lists = []
        self.acc = []
        self.gold_sents = []
        self.prec_1 = []
        self.prec_2 = []
        self.prec_3 = []
        self.prec_4 = []
        self.prec_5 = []
        self.spearman = []
        self.kendalltau = []

    def save_eval_data(self, eval_file = None):
        if eval_file == None:
            # save evaluation output
            if os.path.isdir(self.experiment_file):
                self.experiment_file = self.experiment_file + ".jsonl"

            eval_file = self.experiment_file.replace("outputs", "eval")

        os.makedirs(f"{'/'.join(eval_file.split('/')[:-1])}", exist_ok = True)
        self.eval_data.to_json(eval_file)

    def report_accuracy(self):
        # report mean accuracy
        acc_mean = float(self.eval_data.with_format("pandas")["accuracy"].mean())

        self.logger.run_data.update({"mean_acc": acc_mean})

        if "run_idx" in self.logger.run_data.keys():
            print(f"{self.logger.run_data["run_idx"]}: {acc_mean}")
        else:
            print(f"accuracy: {acc_mean}")
        # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
        #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
        return acc_mean

    def evaluate_sent_order(self):
        """
        This function evaluates the model predictions. 

        @params
            experiment_file: Huggingface folder containing the dataset with the outputs
            self.framework: single/multi-agent setup
            self.logger: experiment logging object 
        @returns
            acc_mean: mean accuracy over all instances
        """
        # evaluate
        if self.framework == "single":
            self.eval_data = self.exp_data.map(self.evaluate_order_answer_single, batched=True, remove_columns=self.exp_data.column_names)
        elif self.framework == "multi":
            self.eval_data = self.exp_data.map(lambda batch: self.evaluate_order_answer_multi(batch), batched=True, remove_columns=self.exp_data.column_names)
        else:
            self.eval_data = self.exp_data.map(self.evaluate_order_answer_random, batched=True, remove_columns=self.exp_data.column_names)
        
        self.save_eval_data()

        acc_mean = self.report_accuracy()
        # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
        #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
        return acc_mean


    def evaluate_order_answer_single(self, batch):
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
        prec_1 = []
        prec_2 = []
        prec_3 = []
        prec_4 = []
        prec_5 = []
        spearman = []
        kendalltau = []

        for i in range(len(batch["storyid"])):
            gold = get_sent_list(batch, i)
            gold_sents.append(gold)

            if "turns" in batch.keys():
                pred = self.extract_answer_list(batch["turns"][i][-1].split("</think>")[-1].strip(), gold)
            else:
                pred = self.extract_answer_list(batch["output"][i].split("</think>")[-1].strip(), gold)
            pred_lists.append(pred)

            acc.append(self.compute_accuracy_order(pred, batch, i))

            prec_n = self.compute_precision(pred, gold)
            prec_1.append(prec_n[0])
            prec_2.append(prec_n[1])
            prec_3.append(prec_n[2])
            prec_4.append(prec_n[3])
            prec_5.append(prec_n[4])

            correlations = self.compute_correlations(pred, gold)
            spearman.append(correlations[0])
            kendalltau.append(correlations[1])
        
        return {"storyid": batch["storyid"], "accuracy": acc, "predictions": pred_lists, "gold": gold_sents, "prec_1": prec_1, "prec_2": prec_2, "prec_3": prec_3, "prec_4": prec_4, "prec_5": prec_5, "spearman": spearman, "kendalltau": kendalltau}

    def evaluate_order_answer_multi(self, batch):
        """
        This function collect the evaluation results for the multi-agent outputs.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        """
        self.initiate_metric_lists()
        for i in range(len(batch["storyid"])):
            gold = get_sent_list(batch, i)
            self.gold_sents.append(gold)

            if "turns" in batch.keys():
                pred = self.extract_answer_list(batch["turns"][i][-1].split("</think>")[-1].strip(), gold) # exclude thinking to retrieve only the final result
            else:
                pred = self.extract_answer_list(batch["output"][i][-1]["answer"].split("</think>")[-1].strip(), gold)
            
            self.pred_lists.append(pred)

            # metrics
            self.acc.append(self.compute_accuracy_order(pred, batch, i))

            prec_n = self.compute_precision(pred, gold)
            self.prec_1.append(prec_n[0])
            self.prec_2.append(prec_n[1])
            self.prec_3.append(prec_n[2])
            self.prec_4.append(prec_n[3])
            self.prec_5.append(prec_n[4])

            correlations = self.compute_correlations(pred, gold)
            self.spearman.append(correlations[0])
            self.kendalltau.append(correlations[1])
        
        return {"storyid": batch["storyid"], "accuracy": self.acc, "predictions": self.pred_lists, "gold": self.gold_sents, "prec_1": self.prec_1, "prec_2": self.prec_2, "prec_3": self.prec_3, "prec_4": self.prec_4, "prec_5": self.prec_5, "spearman": self.spearman, "kendalltau": self.kendalltau}
        
    def evaluate_order_answer_random(self, batch):
        """
        This function collect the evaluation results for the shuffled random outputs.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        """
        pred_lists = []
        acc = []
        gold_sents = []
        prec_1 = []
        prec_2 = []
        prec_3 = []
        prec_4 = []
        prec_5 = []
        spearman = []
        kendalltau = []

        for i in range(len(batch["storyid"])):
            pred = batch["shuffled_sents"][i]
            pred_lists.append(pred)

            gold = get_sent_list(batch, i)
            gold_sents.append(gold)

            acc.append(self.compute_accuracy_order(pred, batch, i))

            prec_n = self.compute_precision(pred, gold)
            prec_1.append(prec_n[0])
            prec_2.append(prec_n[1])
            prec_3.append(prec_n[2])
            prec_4.append(prec_n[3])
            prec_5.append(prec_n[4])

            correlations = self.compute_correlations(pred, gold)
            spearman.append(correlations[0])
            kendalltau.append(correlations[1])
        
        return {"storyid": batch["storyid"], "accuracy": acc, "predictions": pred_lists, "gold": gold_sents, "prec_1": prec_1, "prec_2": prec_2, "prec_3": prec_3, "prec_4": prec_4, "prec_5": prec_5, "spearman": spearman, "kendalltau": kendalltau}

    def extract_answer_list(self, model_out: str, gold: list):
        """
        This function extract the list with predictions from the model output string. 

        @param
            model_out: output of the model
        @returns
            list with the model prediction
        """
        #print("model out", model_out)
        try:
            model_answer = self.clean_out(model_out)
            model_answer = self.clean_match(model_answer)
            #print("model answer", model_answer)
            # model only returned output list
            model_pred = ast.literal_eval(model_answer)
            #print("model pred", model_pred)
            
            if type(model_pred) in (list, tuple):
                return self.clean_preds(model_pred, gold)
            elif type(model_pred) == str:
                # extract list from model output
                str_list = self.match_list(model_pred) # only use first guess by the model

                try:
                    return self.clean_preds(ast.literal_eval(str_list), gold = gold)
                except:
                    return self.clean_preds(self.extract_structured_answer(model_answer), gold)
            else:
                return self.clean_preds(self.extract_structured_answer(model_answer), gold)
        except:
            # extract list from model output
            str_list = self.match_list(model_answer) # only use first guess by the model

            try:
                return self.clean_preds(ast.literal_eval(str_list), gold = gold)
            except:
                return self.clean_preds(self.extract_structured_answer(model_answer), gold)
            
    def extract_structured_answer(self, model_answer):
        conv_str = model_answer.split("</think>")[-1].strip()
        conv_str = re.sub(r"Answer:[ \n]*", "", conv_str).strip()
        # remove newlines to make matching possible
        conv_str = re.sub("\n", "", conv_str)

        # escape single quote strings
        conv_str = re.sub(r'(\b|\b )"(\b| )', r'\1\\"\2', conv_str)
        conv_str = re.sub(r"(\b|\b )'(\b| )", r"\1\\'\2", conv_str)

        # remove json notation
        conv_str = re.sub(r"```json(.*)```", r"\1", conv_str)

        # double quotes
        conv_str = re.sub(r"('|\\\'|\"|\\\"){2,}", r'"', conv_str)
        conv_str = fr"{conv_str}"

        try:
            conv_list = ast.literal_eval(conv_str)
            return conv_list
        except:
            conv_list = []

            sentences = [item[0] for item in re.findall(r"""sentence[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            positions = re.findall(r"""position[\'"]: (\d)""", conv_str)

            for i, sent in enumerate(sentences):
                conv_dict = defaultdict(str)

                conv_dict["sentence"] = sent

                if i < len(positions):
                    conv_dict["positions"] = int(positions[i])
                
                conv_list.append(conv_dict)
        
        return conv_list

    def match_list(self, model_answer: str):
        """
        This function matches a list in the model output.

        @params
            model_answer: model output
        @returns
            str_list: list as a string
        """
        # extract list from model output
        str_match = re.search(r"(\[.*?\])", model_answer) # only use first guess by the model

        if str_match:
            str_list = self.clean_match(str_match.group(1))
        
            return str_list
        else:
            return None

    def clean_out(self, model_out: str):
        """
        This function prepares the model output for the extraction of the prediction list.

        @params
            model_out: full output of the model
        @returns
            model_answer: model answer as single-line string
        """
        model_answer = re.sub(r"Answer:[ \n]*", "", model_out).strip()
        # remove newlines to make matching possible
        model_answer = re.sub("\n", "", model_answer)
        return model_answer

    def clean_match(self, str_list: str):
        """
        This function prepares the extracted string of the prediction list for the transformation into a python list.

        @params
            str_list: string match of the prediction list
        @returns
            str_list: string match with escaped quotation marks and as raw string
        """
        #print("str_list:", str_list)
        # escape single quote strings
        str_list = re.sub(r"(\b|\b )'(\b| )", r"\1\\'\2", str_list)
        str_list = re.sub(r'(\b|\b )"(\b| )', r'\1\\"\2', str_list)

        # remove double quotation marks
        # str_list = re.sub(r"('|\"|\\\"){2,}(.*?)('|\"|\\)+([,\]])", r'"\2"\4', str_list)
        #str_list = re.sub(r'("|\'){2,}', r'"', str_list)
        str_list = re.sub(r"('|\\\'|\"|\\\"){2,}", r'"', str_list)

        # remove escaped brackets
        str_list = re.sub(r"\\{", r'{', str_list)
        str_list = re.sub(r"\\}", r'}', str_list)

        str_list = fr"{str_list}"
        return str_list

    def clean_preds(self, model_pred: list, gold: list):
        """
        This function extracts the predictions from the output list.

        @params
            model_pred: extracted list from the model output
        @returns 
            cleaned_pred: list with model predictions in the correct format
        """
        # get types of items in the list
        pred_types = [(i, type(item)) for i, item in enumerate(model_pred) if type(item) != str]
        cleaned_pred = [""] * 5

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
                elif pred_type == dict and "position" in model_pred[model_i].keys() and "sentence" in model_pred[model_i].keys(): # structured discussions
                    if model_pred[model_i]["position"] != 0:
                        cleaned_pred[model_pred[model_i]["position"]-1] = model_pred[model_i]["sentence"]
                elif pred_type == int:
                    cleaned_pred[model_i] = gold[model_i] # TODO: test
                type_i += 1
        else:
            # turn model prediction into list if the format is already correct
            #print("returning list")
            cleaned_pred = [sent.strip("\"'") for sent in model_pred]

        return cleaned_pred[:5] # only use first 5 sentences

    def compute_accuracy_order(self, pred: list, batch, i: int):
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
                    sent = self.clean_sent(pred[sent_order])
                    if sent == batch[f"sentence{sent_order+1}"][i].lower().strip('.'):
                        acc += 1
            
            return acc/5 # 5 sentences
        else:
            return 0.0
        
    def compute_correlations(self, pred: list, gold: list):
        pred = [self.clean_sent(item) for item in pred]
        gold = [item.lower().strip('.') for item in gold]

        spearman = 0.0 # 0.0 because it indicates no correlation and because Huggingface does not allow differing data formats (e.g., None)
        kendall = 0.0

        gold_ranks = [1, 2, 3, 4, 5]
        pred_ranks = []

        for gold_sent in gold:
            if gold_sent in pred:
                pred_ranks.append(pred.index(gold_sent)+1)

        if len(pred_ranks) == 5:
            spearman = stats.spearmanr(gold_ranks, pred_ranks).statistic
            kendall = stats.kendalltau(gold_ranks, pred_ranks).statistic

        return spearman, kendall

    def clean_sent(self, sent: str):
        sent = sent.lower().strip(".")
        return sent
        
    def compute_precision(self, pred: list, gold: list):
        prec_n = []
        for i in range(5):
            prec_n.append(self.compute_n_precision(pred, gold, i+1))

        return prec_n

    def compute_n_precision(self, pred: list, gold: list, n: int):
        pred = [self.clean_sent(item) for item in pred]
        gold = [item.lower().strip('.') for item in gold]

        correct_ngrams = 0
        gold_ngrams = self.create_gold_ngrams(gold, n)
        for idx in range((len(pred)-n)+1):
            n_pred = pred[idx:idx+n]
            if n_pred in gold_ngrams:
                correct_ngrams += 1 
        
        return correct_ngrams/len(gold_ngrams)

    def create_gold_ngrams(self, gold: list, n: int):
        gold_ngrams = []

        for i in range((len(gold)-n)+1):
            gold_ngrams.append(gold[i:i+n]) 
        
        return gold_ngrams
        
    def compare_instances(self, eval_files: dict, output_file: str = None, insts: int = None):
        """
        This function compares a bunch of experiments against each other.

        @params
            eval_files: dictionary mapping the experiment name to the evaluation file.
        """
        # load datasets
        eval_datasets = defaultdict()
        results = defaultdict(lambda: defaultdict(float))
        metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]
        for exp_name, eval_file in eval_files.items():
            #eval_datasets[exp_name] = load_dataset("json", data_files=eval_file, split="train").select(range(10))
            eval_data = load_dataset("json", data_files=eval_file, split="train").select(range(insts))

            # store mean accuracy per experiment
            for metric in metrics:
                results[exp_name][f"mean_{metric}"] = float(eval_data.with_format("pandas")[metric].mean())

            eval_datasets[exp_name] = eval_data

        # compare evaluation on instance level
        for i in tqdm(range(len(eval_data)), desc="Processing instances", total = len(eval_data)):
            for metric in metrics:
                results = compare_metrics(metric, eval_data, eval_datasets, results, i)

        # save comparison data in csv
        if not output_file:
            output_file = f"{'/'.join(eval_files[exp_name].split('/')[:-1])}/eval_compare.csv"
        with open(output_file, 'w', encoding="utf-8") as csvfile:
            fieldnames = ["experiment_name", "candidates", "experiment_file", "wrong_format"]
            for metric in metrics:
                fieldnames.append(f"mean_{metric}")
                fieldnames.append(f"max_{metric}")
                fieldnames.append(f"min_{metric}")
                fieldnames.append(f"equal_{metric}")

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            
            writer.writeheader()
            for exp, file in eval_files.items():
                writing_dict = defaultdict()
                writing_dict.update({
                    "experiment_name": exp, 
                    "candidates": list(eval_datasets.keys()), 
                    "experiment_file": file, 
                    "wrong_format": results[exp]["wrong_format"]/len(eval_data)
                })

                for metric in metrics:
                    writing_dict[f"mean_{metric}"] =  results[exp][f"mean_{metric}"]
                    writing_dict[f"max_{metric}"] =  results[exp][f"max_{metric}"]/len(eval_data)
                    writing_dict[f"min_{metric}"] =  results[exp][f"min_{metric}"]/len(eval_data)
                    writing_dict[f"equal_{metric}"] =  results[exp][f"equal_{metric}"]/len(eval_data)

                writer.writerow(writing_dict)

    def compare_metrics(self, metric_name: str, eval_data, eval_datasets: defaultdict, results: defaultdict, i: int):
        all_metrics = defaultdict()

        # check if accuracy is the same for all experiments
        same = True
        comp_metric = eval_data[metric_name][i]

        for curr_exp, curr_eval_data in eval_datasets.items():
            curr_metric = curr_eval_data[metric_name][i]
            all_metrics[curr_exp] = curr_metric

            if curr_metric != comp_metric:
                same = False # accuracy between experiments differs

            if metric_name == "accuracy" and curr_eval_data["predictions"][i] == ["", "", "", "", ""]:
                # check whether model returned prediction in the correct format
                results[curr_exp]["wrong_format"] += 1

        # compare accuracies between experiments
        for curr_exp in eval_datasets.keys():
            if max(all_metrics.values()) == all_metrics[curr_exp]:
                results[curr_exp][f"max_{metric_name}"] += 1 # experiment has the best accuracy

            if min(all_metrics.values()) == all_metrics[curr_exp]:
                results[curr_exp][f"min_{metric_name}"] += 1 # experiment has the worst accuracy

            if same:
                results[curr_exp][f"equal_{metric_name}"] += 1 # instance is not affected by experiment
            
        return results

    def evaluate_all_outputs(self, splitdir = "outputs/single/Llama-3.2-1B-Instruct/dev/"):
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
                self.experiment_file = splitdir + outdir
                self.exp_data = load_from_disk(self.experiment_file)
                if self.last:
                    print(f"\n{outdir}: {self.evaluate_last_beliefs()}\n")
                else:
                    print(f"\n{outdir}: {self.evaluate_sent_order()}\n")

        #print(evals)

    def create_compare_dict(self, split):
        df = pd.read_csv("./logs/runs.csv", sep = ";")

        filtered_df = df
        
        if self.framework != None:
            filtered_df = filtered_df.loc[df["self.framework"] == self.framework]
        
        if split != None: 
            filtered_df = filtered_df.loc[df["split"] == split]

        files = filtered_df["experiment_file"].dropna()
        min_insts = None
        run2files = defaultdict(str)

        for file in files:
            if not file.endswith(".jsonl") and "ckps" not in file:
                run_data = df.loc[df["experiment_file"] == file]
                run2files[int(run_data["run_idx"].values[0])] = f"{file.replace('outputs', 'eval')}.jsonl"

                insts = int(run_data["instances"].values[0])
                if min_insts == None or min_insts > insts:
                    min_insts = insts
        
        return run2files, min_insts

    def update_evaluation_files(self, frameworks = ["single", "multi"], splits = ["mini", "dev", "test"]):
        for fw in frameworks: 
            self.framework = fw
            models = os.listdir(f"outputs/{fw}/")

            for model in models:
                for split in splits: 
                    dir_path = f"outputs/{fw}/{model}/{split}/"
                    if os.path.isdir(dir_path):
                        self.evaluate_all_outputs(dir_path)

    def create_belief_instance(self, out, eval):
        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)

                player = out["player"][i]

                for conv_dir in conv_list:
                    if type(conv_dir) == dict and conv_dir["sentence"] in sent2gold.keys() and "position" in conv_dir.keys():
                        sent_idx = sent2gold[conv_dir["sentence"]]
                        
                        # Player, sentence, position, time-step
                        if sent_idx == conv_dir["position"]:
                            correct = "correct"
                        else:
                            correct = "incorrect"

                        beliefs.append({
                            "sentence": conv_dir["sentence"],
                            "turn": i+1,
                            "agent": player,
                            "gold_sent_idx": sent_idx,
                            "hypo": conv_dir["position"],
                            "correct": correct,
                            "label": f"({player}, sentence {sent_idx}, hypothesis {conv_dir["position"]})"
                        })

        return {"beliefs": beliefs}

    def evaluate_last_beliefs(self):
        eval_file = f"{self.experiment_file.replace("outputs", "eval").split(".")[0]}.jsonl"

        full_eval_data = self.load_eval_data(eval_file)

        exp_beliefs = self.exp_data.map(lambda x, idx: self.create_belief_instance(x, full_eval_data.select([idx])[0]), batched = False, with_indices=True)

        preds = exp_beliefs.map(self.create_pred_from_belief, batched = False)

        self.eval_data = preds.map(lambda batch: self.calculate_metrics(batch), batched = True)

        self.save_eval_data(f"{self.experiment_file.replace("outputs", "eval")}_last.jsonl")

        acc_mean = self.report_accuracy()
        # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
        #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
        return acc_mean


    def create_pred_from_belief(self, beliefs):
        # list of beliefs with dict, describing the belief
        pred = [""]*5

        for belief in beliefs["beliefs"][::-1]:
            if belief["agent"] != "Player 6" and belief["sentence"] not in pred:
                pred[belief["hypo"]-1] = belief["sentence"]

        return {"pred": pred}
    
    def calculate_metrics(self, batch):
        self.initiate_metric_lists()
        for i in range(len(batch["storyid"])):
            self.pred_lists = list(batch["pred"])

            gold = get_sent_list(batch, i)
            self.gold_sents.append(gold)

            self.acc.append(self.compute_accuracy_order(batch["pred"][i], batch, i))

            prec_n = self.compute_precision(batch["pred"][i], gold)
            self.prec_1.append(prec_n[0])
            self.prec_2.append(prec_n[1])
            self.prec_3.append(prec_n[2])
            self.prec_4.append(prec_n[3])
            self.prec_5.append(prec_n[4])

            correlations = self.compute_correlations(batch["pred"][i], gold)
            self.spearman.append(correlations[0])
            self.kendalltau.append(correlations[1])
        
        return {"storyid": batch["storyid"], "accuracy": self.acc, "predictions": self.pred_lists, "gold": self.gold_sents, "prec_1": self.prec_1, "prec_2": self.prec_2, "prec_3": self.prec_3, "prec_4": self.prec_4, "prec_5": self.prec_5, "spearman": self.spearman, "kendalltau": self.kendalltau}

    def load_eval_data(self, eval_file):
        try:
            return load_dataset("json", data_files=eval_file, split="train")
        except:
            self.evaluate_sent_order()
            return load_dataset("json", data_files=eval_file, split="train")

if __name__ == "__main__":
    cli = CLI()

    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-14_10:24:56.857093.jsonl"
    #outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_08:48:25.663742"

    # outfile = "outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881"
    # #evaluate_sent_order("test.jsonl", "single")
    #evaluate_sent_order("./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:22:00.500021", "single")

    #outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-17_13:10:53.603867"
    #outdir = "./outputs/single/Llama-3.2-1B-Instruct/dev/outputs_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743"
    #outdir = "./outputs/multi/Qwen3-8B/dev/outputs_multi_Qwen3-8B_dev_2025-11-25_13:47:22.215646" # 73
    outdir = "outputs/multi/Qwen3-32B/dev/outputs_154"
    #evaluate_sent_order(outdir, "multi", turn_idx=-2)

    evaluator = Evaluator(outdir, "multi", last = True)
    evaluator.update_evaluation_files(frameworks=["multi"], splits = ["dev"])
    #evaluator.evaluate_sent_order(turn_idx = -2)
    #evaluator.evaluate_last_beliefs()

    # evaluate_all_outputs(splitdir="outputs/multi/Qwen3-8B/dev/", self.framework="multi")
    # evaluate_all_outputs(splitdir="outputs/multi/Llama-3.2-1B-Instruct/dev/", self.framework="multi")
    # evaluate_all_outputs(splitdir="outputs/multi/Qwen3-8B/dev/", self.framework="multi")

    # prompt_dev_files = {
    #     "self": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881.jsonl",
    #     "gemini": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:22:00.500021.jsonl",
    #     "llama": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743.jsonl"
    # }
    # compare_instances(prompt_dev_files)

    # prompt_dev_files_multi = {
    #     "self": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-17_11:52:54.422881.jsonl",
    #     "gemini": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:22:00.500021.jsonl",
    #     "llama": "./eval/single/Llama-3.2-1B-Instruct/dev/eval_single_Llama-3.2-1B-Instruct_dev_2025-11-19_09:17:22.874743.jsonl"
    # }
    #compare_instance_acc(prompt_dev_files_multi)

    # dev_multi_models = {
    #     "Llama-3.2-1B-Instruct_dev_2025-11-17_13:45:49.929179": "eval/multi/Llama-3.2-1B-Instruct/dev/eval_multi_Llama-3.2-1B-Instruct_dev_2025-11-17_13:45:49.929179.jsonl",
    #     "Llama-3.2-1B-Instruct_dev_2025-11-17_17:55:13.130931": "eval/multi/Llama-3.2-1B-Instruct/dev/eval_multi_Llama-3.2-1B-Instruct_dev_2025-11-17_17:55:13.130931.jsonl",
    #     "Llama-3.2-1B-Instruct_dev_2025-11-18_00:13:58.547826": "eval/multi/Llama-3.2-1B-Instruct/dev/eval_multi_Llama-3.2-1B-Instruct_dev_2025-11-18_00:13:58.547826.jsonl",
    #     "Llama-3.2-1B-Instruct_dev_2025-11-18_11:43:21.920146": "eval/multi/Llama-3.2-1B-Instruct/dev/eval_multi_Llama-3.2-1B-Instruct_dev_2025-11-18_11:43:21.920146.jsonl",
    #     "73": "eval/multi/Qwen3-8B/dev/eval_multi_Qwen3-8B_dev_2025-11-25_13:47:22.215646.jsonl",
    #     "75": "eval/multi/Qwen3-14B/dev/eval_75.jsonl",
    #     "81": "eval/multi/Qwen3-8B/dev/eval_81.jsonl",
    # } 

    #update_evaluation_files(self.frameworks=["multi"], splits=["dev"])
    # dev_multi_models, insts = create_compare_dict(self.framework="multi", split="dev")
    # compare_instances(dev_multi_models, "./eval/multi/eval_compare_models_dev.csv", insts)

