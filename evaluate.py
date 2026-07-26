from datasets import load_dataset, load_from_disk
import ast
from data import get_sent_list
import os
import re
import pandas as pd
from log import ExperimentLog
import os
from collections import defaultdict
import csv
from tqdm import tqdm
from scipy import stats
import random

def evaluate(outfile, framework, task: str, logger: ExperimentLog = ExperimentLog(), structured: bool = False, sanity_checks: bool = False, all: bool = False, instances: int = None, last: bool = False):
    '''
    This function evaluates the model predictions for a given experiment output file.

    @params
        outfile: path to the output dataset
        framework: single/multi-agent setup
        task: rocstories/cmu_booksum
        logger: experiment logging object
        structured: whether the model output is structured or not
        sanity_checks: whether to return the number of wrong format instances or not
        all: whether to return all evaluation results or just the mean accuracy
    @returns
        results: dictionary with the evaluation results/mean accuracy, or the number of wrong format instances if sanity_checks is True
    '''
    evaluator = Evaluator(experiment_file= outfile, framework= framework, logger = logger, last = False, task = task, instances = instances)

    results = defaultdict(float)

    if task == "rocstories":
        results["acc_mean"] = evaluator.evaluate_sent_order()

        results["prec_1"] = float(evaluator.eval_data.with_format("pandas")["prec_1"].mean())
        results["prec_2"] = float(evaluator.eval_data.with_format("pandas")["prec_2"].mean())
        results["prec_3"] = float(evaluator.eval_data.with_format("pandas")["prec_3"].mean())
        results["prec_4"] = float(evaluator.eval_data.with_format("pandas")["prec_4"].mean())
        results["prec_5"] = float(evaluator.eval_data.with_format("pandas")["prec_5"].mean())
        results["spearman"] = float(evaluator.eval_data.with_format("pandas")["spearman"].mean())
        results["kendalltau"] = float(evaluator.eval_data.with_format("pandas")["kendalltau"].mean())
    else:
        results["acc_mean"] = evaluator.evaluate_odd_one()

    results["instances"] = instances
    results["wrong_format"] = evaluator.wrong_format
    results["zero_counter"] = evaluator.zero_counter
    results["zero_percentage"] = evaluator.zero_percentage
    results["double_counter"] = evaluator.double_counter
    results["double_percentage"] = evaluator.double_percentage

    evaluator.last = True
    evaluator.zero_counter = 0
    evaluator.double_counter = 0
    evaluator.wrong_format = 0

    if structured == True and last == True:
        results["acc_last"] = evaluator.evaluate_last_beliefs()

        results["wrong_format_last"] = evaluator.wrong_format
        results["zero_percentage_last"] = evaluator.zero_percentage
        results["double_percentage_last"] = evaluator.double_percentage

        if task == "rocstories":
            results["prec_1_last"] = float(evaluator.eval_data.with_format("pandas")["prec_1"].mean())
            results["prec_2_last"] = float(evaluator.eval_data.with_format("pandas")["prec_2"].mean())
            results["prec_3_last"] = float(evaluator.eval_data.with_format("pandas")["prec_3"].mean())
            results["prec_4_last"] = float(evaluator.eval_data.with_format("pandas")["prec_4"].mean())
            results["prec_5_last"] = float(evaluator.eval_data.with_format("pandas")["prec_5"].mean())
            results["spearman_last"] = float(evaluator.eval_data.with_format("pandas")["spearman"].mean())
            results["kendalltau_last"] = float(evaluator.eval_data.with_format("pandas")["kendalltau"].mean())

    if all:
        return results
    elif sanity_checks:
        return results["wrong_format"], results["wrong_format_last"]
    else:
        return results["acc_mean"], results["acc_last"]

class Evaluator:
    def __init__(self, experiment_file: str, framework: str, task: str, logger: ExperimentLog = ExperimentLog(), idx = None, last = False, instances = None):
        '''
        This class evaluates the model predictions for a given experiment output file.

        @params
            experiment_file: path to the output dataset
            framework: single/multi-agent setup
            task: rocstories/cmu_booksum
            logger: experiment logging object
            idx: index of the instance to evaluate, if None evaluates all instances
            last: whether to evaluate the last beliefs or not
            instances: number of instances to evaluate, if None evaluates all instances
        '''
        self.experiment_file = experiment_file
        if idx:
            self.exp_data = load_from_disk(experiment_file).select([idx])
        else:
            self.exp_data = load_from_disk(experiment_file)

        if instances:
            self.exp_data = self.exp_data.select(range(instances))

        self.framework = framework
        self.task = task
        self.logger = logger

        self.initiate_metric_lists()

        self.last = last

        # check evaluation results
        self.zero_counter = 0
        self.double_counter = 0
        self.wrong_format = 0

        if self.task == "rocstories":
            self.pred_len = 5
        else:
            self.pred_len = 3
        
    def initiate_metric_lists(self):
        self.pred_lists = []
        self.acc = []

        if self.task == "rocstories":
            self.gold_sents = []
            self.prec_1 = []
            self.prec_2 = []
            self.prec_3 = []
            self.prec_4 = []
            self.prec_5 = []
            self.spearman = []
            self.kendalltau = []

    def save_eval_data(self, eval_file = None):
        '''
        This method saves the evaluation results to a jsonl file.

        @params
            eval_file: path to the evaluation file, if None saves to the default location
        '''
        if eval_file == None:
            # save evaluation output
            if os.path.isdir(self.experiment_file):
                self.experiment_file = self.experiment_file + ".jsonl"

            eval_file = self.experiment_file.replace("outputs", "eval")

        os.makedirs(f"{'/'.join(eval_file.split('/')[:-1])}", exist_ok = True)
        self.eval_data.to_json(eval_file)

    def report_accuracy(self):
        '''
        This method reports the mean accuracy of the evaluation results and logs it to the experiment logger.

        @returns
            acc_mean: mean accuracy over all instances
        '''
        # report mean accuracy
        acc_mean = float(self.eval_data.with_format("pandas")["accuracy"].mean())

        if self.last:
            self.logger.run_data.update({"last_mean_acc": acc_mean})
            self.logger.run.log({"last_mean_acc": acc_mean})
        else:
            self.logger.run_data.update({"mean_acc": acc_mean})
            self.logger.run.log({"mean_acc": acc_mean})

        if "run_idx" in self.logger.run_data.keys():
            print(f"{self.logger.run_data['run_idx']}: {acc_mean}")
        else:
            print(self.experiment_file)
            print(f"accuracy: {acc_mean}")

        self.zero_percentage = self.zero_counter/(len(self.eval_data)*self.pred_len)
        print("zero percentage:", self.zero_percentage)
        self.logger.run_data.update({"zero_percentage": self.zero_percentage})
        self.logger.run.log({"zero_percentage": self.zero_percentage})

        self.double_percentage = self.double_counter/(len(self.eval_data)*self.pred_len)
        print("double percentage:", self.double_percentage)
        self.logger.run_data.update({"double_percentage": self.double_percentage})
        self.logger.run.log({"double_percentage": self.double_percentage})

        self.wrong_format = self.wrong_format/len(self.eval_data) # for not-last, there is one chance per instance
        print("wrong format:", self.wrong_format)
        self.logger.run_data.update({"wrong_format": self.wrong_format})
        self.logger.run.log({"wrong_format": self.wrong_format})

        return acc_mean

    def evaluate_sent_order(self):
        """
        This function evaluates the model predictions for ROCStories. 

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
        return acc_mean

    def evaluate_odd_one(self):
        """
        This function evaluates the model predictions for CMU Booksum. 

        @returns
            acc_mean: mean accuracy over all instances
        """
        if self.framework == "single":
            self.eval_data = self.exp_data.map(self.evaluate_odd_answer_single, batched=True, remove_columns=self.exp_data.column_names)
        elif self.framework == "multi":
            self.eval_data = self.exp_data.map(lambda batch: self.evaluate_odd_answer_multi(batch), batched=True, remove_columns=self.exp_data.column_names)
        else:
            self.eval_data = self.exp_data.map(self.evaluate_odd_answer_random, batched=True, remove_columns=self.exp_data.column_names)
        
        self.save_eval_data()

        acc_mean = self.report_accuracy()
        return acc_mean

    def evaluate_odd_answer_single(self, batch):
        '''
        This method collects the evaluation results for the single-agent outputs for the CMU Booksum task.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        '''
        pred_lists = []
        acc = []
        gold_lbls = []

        for i in range(len(batch["id"])):
            if "gold_title" in batch.keys():
                gold = batch["gold_title"][i]
            else:
                gold = [batch["titles"][i][idx] for idx, lbl in enumerate(batch["gold_lbls"][i]) if lbl == 1][0]

            gold_lbls.append(gold)

            if "turns" in batch.keys():
                pred = self.extract_answer_list(batch["turns"][i][-1].split("</think>")[-1].strip(), gold)
            else:
                pred = self.extract_answer_list(batch["output"][i].split("</think>")[-1].strip(), gold)
            
            pred = self.extract_pred_odd(pred, batch["titles"][i])
            pred_lists.append(pred)
        
            acc.append(self.compute_accuracy_odd(pred, gold, batch["titles"][i].index(gold)+1))
        
        return {"id": batch["id"], "accuracy": acc, "predictions": pred_lists, "gold": gold_lbls}


    def evaluate_order_answer_single(self, batch):
        """
        This function collect the evaluation results for the single-agent outputs for ROCStories.

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
        This function collect the evaluation results for the multi-agent outputs for ROCStories.

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
        
    def evaluate_odd_answer_multi(self, batch):
        '''
        This method collects the evaluation results for the multi-agent outputs for the CMU Booksum task.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        '''
        self.initiate_metric_lists()
        for i in range(len(batch["id"])):
            gold = batch["gold_title"][i]

            if "turns" in batch.keys():
                out = batch["turns"][i][-1].split("</think>")[-1].strip() # exclude thinking to retrieve only the final result
            else:
                out = batch["output"][i][-1]["answer"].split("</think>")[-1].strip()

            pred = self.extract_pred_odd(out, batch["titles"][i])
            self.pred_lists.append(pred)

            # metrics
            self.acc.append(self.compute_accuracy_odd(pred, gold, batch["titles"][i].index(gold)+1))
        
        return {"id": batch["id"], "accuracy": self.acc, "predictions": self.pred_lists, "gold": list(batch["gold_title"])}
        
    def extract_pred_odd(self, pred, titles):
        '''
        This method extracts the predicted title from the model output for the CMU Booksum task.

        @param
            pred: model output
            titles: list of possible titles
        @returns
            string with the predicted title
        '''
        if type(pred) == str:
            match = re.search(r"different genre is:\n\n([\w *]+)", pred)
            if match:
                return match.group(1).strip()
            else:
                pred = pred.replace("\\'", "'")
                pred = pred.replace('\\"', '"')

                max_id = 0
                max_title = ""
                for title in titles:
                    subtitles = title.split(":")

                    for subtitle in subtitles:
                        # TODO: "Ronia the Robber\\'s Daughter" == "Ronia the Robber's Daughter"
                        subtitle = subtitle.strip().lower()
                        if subtitle.lower() in pred.lower():
                            curr_idx = pred.lower().rfind(subtitle.lower()) # get last title occurrence as answer

                            if curr_idx >= max_id:
                                max_title = title
                                max_id = curr_idx
                
                if max_title == "":
                    self.wrong_format += 1 # Sanity check for missing titles

                return max_title
        else:
            return [titles[idx] for idx, lbl in enumerate(pred) if lbl == 1][0]

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

    def evaluate_odd_answer_random(self, batch):
        '''
        This method collects the evaluation results for the random outputs for the CMU Booksum task.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        '''
        acc = []
        pred_lists = []

        for i in range(len(batch["id"])):
            rand_idx = random.randint(0, 2)
            pred = batch["titles"][i][rand_idx]
                        
            pred_lists.append(pred)

            gold = batch["gold_title"][i]
            acc.append(self.compute_accuracy_odd(pred, gold, batch["titles"][i].index(gold)+1))

        return {"id": batch["id"], "accuracy": acc, "predictions": pred_lists, "gold": list(batch["gold_title"])}

    def extract_answer_list(self, model_out: str, gold: list):
        """
        This function extract the list with predictions from the model output string. 

        @param
            model_out: output of the model
        @returns
            list with the model prediction
        """
        try:
            model_answer = self.clean_out(model_out)
            model_answer = self.clean_match(model_answer)
            # model only returned output list
            model_pred = ast.literal_eval(model_answer)
            
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
            if self.task == "rocstories":
                str_list = self.match_list(model_answer) # only use first guess by the model

                try:
                    return self.clean_preds(ast.literal_eval(str_list), gold = gold)
                except:
                    return self.clean_preds(self.extract_structured_answer(model_answer), gold)
            else:
                return model_answer
            
    def extract_structured_answer(self, model_answer):
        '''
        This method extracts the structured answer from the model output for both ROCStories and CMU Booksum tasks.

        @param
            model_answer: model output
        @returns
            conv_list: list with the model predictions in the correct format
        '''
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
        
        if self.task == "rocstories":
            cleaned_pred = [""] * self.pred_len
        else:
            cleaned_pred = [-1] * self.pred_len

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
                elif pred_type in [dict, defaultdict] and "position" in model_pred[model_i].keys() and "sentence" in model_pred[model_i].keys(): # structured discussions
                    hypo = int(model_pred[model_i]["position"])
                    
                    if hypo <= len(cleaned_pred): 
                        if cleaned_pred[hypo-1] != "":
                            self.double_counter += 1

                        if hypo != 0:
                            cleaned_pred[hypo-1] = model_pred[model_i]["sentence"]
                        else:
                            self.zero_counter += 1
                elif pred_type == int:
                    cleaned_pred[model_i] = gold[model_i] # TODO: test
                elif pred_type in [dict, defaultdict] and "label" in model_pred[model_i].keys() and "title" in model_pred[model_i].keys():
                    hypo = int(model_pred[model_i]["label"])

                    if hypo in [0, 1]:
                        cleaned_pred[model_i] = hypo
                    else:
                        print("Found instance with wrong label")
                type_i += 1
        else:
            # turn model prediction into list if the format is already correct
            #print("returning list")
            cleaned_pred = [sent.strip("\"'") for sent in model_pred]

        return cleaned_pred[:self.pred_len] # only use first 5 sentences

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
            
            return acc/self.pred_len # 5 sentences
        else:
            return 0.0
        
    def compute_accuracy_odd(self, pred: str, gold_title: str, gold_title_id: str):
        num_correct = 0
        subtitles = gold_title.split(":")

        for subtitle in subtitles:
            subtitle = subtitle.strip().lower()
            if subtitle in pred.lower() or str(gold_title_id) in pred.lower():
                return 1
            # else:
            #     print("incorrect prediction")
        
        return num_correct
        
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
        """
        @deprecated?
        """
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

                if "rocstories" in self.experiment_file:
                    acc_mean = self.evaluate_sent_order()
                else:
                    acc_mean = self.evaluate_odd_one()

                self.last = True
                self.zero_counter = 0
                self.double_counter = 0
                self.wrong_format = 0

                acc_last = 0.0

                if self.framework == "multi":
                    self.evaluate_last_beliefs()

                
                # self.exp_data = load_from_disk(self.experiment_file)
                # if self.last:
                #     print(f"\n{outdir}: {self.evaluate_last_beliefs()}\n")
                # else:
                #     print(f"\n{outdir}: {self.evaluate_sent_order()}\n")

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

    def update_evaluation_files(self, frameworks = ["single", "multi"], splits = ["mini", "dev", "test"], tasks = ["rocstories", "cmu_booksum", "booksum"]):
        for fw in frameworks: 
            self.framework = fw
            models = os.listdir(f"outputs/{fw}/")

            for model in models:
                for task in tasks:
                    self.task = task
                    for split in splits: 
                        self.split = split
                        dir_path = f"outputs/{fw}/{model}/{task}/{split}/"
                        if os.path.isdir(dir_path):
                            self.evaluate_all_outputs(dir_path)

    def create_belief_instance_roc(self, out, eval):
        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)
                
                # format as list not as dict
                if str in [type(item) for item in conv_list]:
                    pred = conv_list.copy()
                    conv_list = []

                    for pos, sent in enumerate(pred, start=1):
                        if sent != "":
                            conv_list.append({"sentence": sent, "position": pos})

                player = out["player"][i]

                if i == (len(out["turns"])-1) and conv_list == []:
                    pred = self.extract_answer_list(turn, eval["gold"])

                    for pos, sent in enumerate(pred, start=1):
                        if sent != "":
                            conv_list.append({"sentence": sent, "position": pos})

                if type(conv_list) == list:
                    # check for 0:
                    zero_count = True in [0 in conv_dir.values() for conv_dir in conv_list]

                    for conv_dir in conv_list:
                        if type(conv_dir) in [defaultdict, dict] and conv_dir["sentence"] in sent2gold.keys() and "position" in conv_dir.keys() and type(conv_dir["position"])==int:
                            if zero_count:
                                pos = conv_dir["position"]+1
                            else:
                                pos = conv_dir["position"]
                            
                            sent_idx = sent2gold[conv_dir["sentence"]]
                            
                            # Player, sentence, position, time-step
                            if sent_idx == pos:
                                correct = "correct"
                            else:
                                correct = "incorrect"

                            beliefs.append({
                                "sentence": conv_dir["sentence"],
                                "turn": i+1,
                                "agent": player,
                                "gold_sent_idx": sent_idx,
                                "hypo": int(conv_dir["position"]),
                                "correct": correct,
                                "label": f"({player}, sentence {sent_idx}, hypothesis {conv_dir['position']})"
                            })

        return {"beliefs": beliefs}
    
    def create_belief_instance_cmu(self, out, eval):
        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)

                if i == (len(out["turns"])-1) and conv_list == []:
                    pred = self.extract_answer_list(turn, eval["gold"])
                    pred = self.extract_pred_odd(pred, out["titles"])

                    conv_list.append({"title": pred, "label": "true"})

                player = out["player"][i]

                if type(conv_list) == list:
                    for conv_dir in conv_list:
                        if type(conv_dir) in [defaultdict, dict] and conv_dir["title"] in out["titles"] and "label" in conv_dir.keys():                        
                            # Player, sentence, position, time-step
                            if "true" in conv_dir["label"].lower() and conv_dir["title"] == eval["gold"]:
                                correct = "correct"
                            elif "false" in conv_dir["label"].lower() and conv_dir["title"] != eval["gold"]:
                                correct = "correct"
                            else:
                                correct = "incorrect"

                            beliefs.append({
                                "title": conv_dir["title"],
                                "turn": i+1,
                                "agent": player,
                                "gold_title": eval["gold"],
                                "hypo": conv_dir["label"],
                                "correct": correct,
                                "label": f"({player}, title {conv_dir['title']}, hypothesis {conv_dir['label']})"
                            })

        return {"beliefs": beliefs}

    def evaluate_last_beliefs(self):
        eval_file = f"{self.experiment_file.replace('outputs', 'eval').split('.jsonl')[0]}.jsonl"

        full_eval_data = self.load_eval_data(eval_file)

        if self.task == "rocstories":
            exp_beliefs = self.exp_data.map(lambda x, idx: self.create_belief_instance_roc(x, full_eval_data.select([idx])[0]), batched = False, with_indices=True)
            preds = exp_beliefs.map(self.create_pred_from_belief_roc, batched = False)
            self.eval_data = preds.map(lambda batch: self.calculate_metrics_roc(batch), batched = True)
        else:
            exp_beliefs = self.exp_data.map(lambda x, idx: self.create_belief_instance_cmu(x, full_eval_data.select([idx])[0]), batched = False, with_indices=True)
            preds = exp_beliefs.map(self.create_pred_from_belief_cmu, batched = False)
            self.eval_data = preds.map(lambda batch: self.calculate_metrics_cmu(batch), batched = True)
        
        self.save_eval_data(f"{eval_file.split('.jsonl')[0]}_last.jsonl")

        acc_mean = self.report_accuracy()
        # with open("eval/acc_log.jsonl", "a", encoding="utf-8") as f:
        #     json.dump({eval_file.split("/")[-1]: acc_mean}, f, indent=4)
        return acc_mean


    def create_pred_from_belief_roc(self, beliefs):
        # list of beliefs with dict, describing the belief
        pred = [""]*5

        for belief in beliefs["beliefs"][::-1]:
            if belief["agent"] not in ["Player 6", "expert"] and belief["sentence"] not in pred and int(belief["hypo"]) < len(pred):
                pred[int(belief["hypo"])-1] = belief["sentence"]

        return {"pred": pred}
    
    def create_pred_from_belief_cmu(self, beliefs):
        # list of beliefs with dict, describing the belief
        pred = ""

        for belief in beliefs["beliefs"][::-1]:
            if belief["agent"] not in ["Player 6", "expert"] and belief["hypo"].lower() == "true":
                pred = belief["title"]

        return {"pred": pred}
    
    def calculate_metrics_roc(self, batch):
        self.initiate_metric_lists()
        self.pred_lists = list(batch["pred"])
        for i in range(len(batch["storyid"])):

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

    def calculate_metrics_cmu(self, batch):
        self.initiate_metric_lists()
        self.pred_lists = list(batch["pred"])
        for i in range(len(batch["id"])):
            gold = batch["gold_title"][i]
            self.acc.append(self.compute_accuracy_odd(batch["pred"][i], gold, batch["titles"][i].index(gold)+1))
        
        return {"id": batch["id"], "accuracy": self.acc, "predictions": self.pred_lists, "gold": list(batch["gold_title"])}


    def load_eval_data(self, eval_file):
        try:
            conv_list = ast.literal_eval(conv_str)
            return conv_list
        except:
            conv_list = []

            if self.task == "rocstories":
                sentences = [item[0] for item in re.findall(r"""sentence[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
                positions = re.findall(r"""position[\'"]: (\d)""", conv_str)

                for i, sent in enumerate(sentences):
                    conv_dict = defaultdict(str)

                    conv_dict["sentence"] = sent

                    if i < len(positions):
                        conv_dict["position"] = int(positions[i])
                    
                    conv_list.append(conv_dict)
            else:
                titles = [item[0] for item in re.findall(r"""title[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
                labels = [item[0] for item in re.findall(r"""label[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]

                for i, title in enumerate(titles):
                    conv_dict = defaultdict(str)

                    conv_dict["title"] = title

                    if i < len(labels):
                        conv_dict["label"] = labels[i]
                    
                    conv_list.append(conv_dict)

        if conv_list == []:
            self.wrong_format += 1
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
        # escape single quote strings
        str_list = re.sub(r"(\b|\b )'(\b| )", r"\1\\'\2", str_list)
        str_list = re.sub(r'(\b|\b )"(\b| )', r'\1\\"\2', str_list)

        # remove double quotation marks
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
        
        if self.task == "rocstories":
            cleaned_pred = [""] * self.pred_len
        else:
            cleaned_pred = [-1] * self.pred_len

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
                elif pred_type in [dict, defaultdict] and "position" in model_pred[model_i].keys() and "sentence" in model_pred[model_i].keys(): # structured discussions
                    hypo = int(model_pred[model_i]["position"])
                    
                    if hypo <= len(cleaned_pred): 
                        if cleaned_pred[hypo-1] != "":
                            self.double_counter += 1

                        if hypo != 0:
                            cleaned_pred[hypo-1] = model_pred[model_i]["sentence"]
                        else:
                            self.zero_counter += 1
                elif pred_type == int:
                    cleaned_pred[model_i] = gold[model_i] # TODO: test
                elif pred_type in [dict, defaultdict] and "label" in model_pred[model_i].keys() and "title" in model_pred[model_i].keys():
                    hypo = int(model_pred[model_i]["label"])

                    if hypo in [0, 1]:
                        cleaned_pred[model_i] = hypo
                    else:
                        print("Found instance with wrong label")
                type_i += 1
        else:
            # turn model prediction into list if the format is already correct
            cleaned_pred = [sent.strip("\"'") for sent in model_pred]

        return cleaned_pred[:self.pred_len] # only use first 5 sentences

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
            
            return acc/self.pred_len # 5 sentences
        else:
            return 0.0
        
    def compute_accuracy_odd(self, pred: str, gold_title: str, gold_title_id: str):
        '''
        This method compares the predicted title with the gold standard title for the CMU Booksum task.

        @param
            pred: model prediction
            gold_title: gold standard title
            gold_title_id: index of the gold standard title in the list of possible titles
        @returns
            accuracy of the model prediction (1 if correct, 0 if incorrect)
        '''
        num_correct = 0
        subtitles = gold_title.split(":")

        for subtitle in subtitles:
            subtitle = subtitle.strip().lower()
            if subtitle in pred.lower() or str(gold_title_id) in pred.lower():
                return 1
        
        return num_correct
        
    def compute_correlations(self, pred: list, gold: list):
        '''
        This method computes the Spearman and Kendall correlation coefficients between the predicted and gold standard sentence orders for the ROCStories task.

        @param
            pred: model prediction
            gold: gold standard sentence order
        @returns
            spearman: Spearman correlation coefficient
            kendall: Kendall correlation coefficient
        '''
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
        '''
        This method cleans the sentence by converting it to lowercase and stripping any trailing periods.

        @param
            sent: sentence to be cleaned
        @returns
            cleaned sentence
        '''
        sent = sent.lower().strip(".")
        return sent
        
    def compute_precision(self, pred: list, gold: list):
        '''
        This method computes the precision of the model predictions compared to the gold standard for ROCStories.

        @param
            pred: model prediction
            gold: gold standard sentence order
        @returns
            precision: list of precision scores for each n (1-5)
        '''
        prec_n = []
        for i in range(5):
            prec_n.append(self.compute_n_precision(pred, gold, i+1))

        return prec_n

    def compute_n_precision(self, pred: list, gold: list, n: int):
        '''
        This method computes the n-gram precision of the model predictions compared to the gold standard for ROCStories.

        @param
            pred: model prediction
            gold: gold standard sentence order
            n: n-gram size
        @returns
            n-gram precision score
        '''
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
        '''
        This method creates a list of n-grams from the gold standard sentence order for ROCStories.

        @param
            gold: gold standard sentence order
            n: n-gram size
        @returns
            gold_ngrams: list of n-grams from the gold standard
        '''
        gold_ngrams = []

        for i in range((len(gold)-n)+1):
            gold_ngrams.append(gold[i:i+n]) 
        
        return gold_ngrams

    def evaluate_all_outputs(self, splitdir = "outputs/single/Llama-3.2-1B-Instruct/dev/"):
        """
        This function evaluates all outputs in a directory.

        @params
            splitdir: directory with the output files
        @overwrites all current evaluation files
        """
        outdirs = os.listdir(splitdir)

        for outdir in outdirs:
            if outdir != "ckps" and not outdir.endswith("jsonl"):
                self.experiment_file = splitdir + outdir
                
                self.exp_data = load_from_disk(self.experiment_file)

                if "rocstories" in self.experiment_file:
                    self.evaluate_sent_order()
                else:
                    self.evaluate_odd_one()

                self.last = True
                self.zero_counter = 0
                self.double_counter = 0
                self.wrong_format = 0

                if self.framework == "multi":
                    self.evaluate_last_beliefs()

    def update_evaluation_files(self, frameworks = ["single", "multi"], splits = ["mini", "dev", "test"], tasks = ["rocstories", "cmu_booksum", "booksum"]):
        '''
        This method updates the evaluation files for all frameworks, splits, and tasks.

        @params
            frameworks: list of frameworks to evaluate
            splits: list of data splits to evaluate
            tasks: list of tasks to evaluate
        '''
        for fw in frameworks: 
            self.framework = fw
            models = os.listdir(f"outputs/{fw}/")

            for model in models:
                for task in tasks:
                    self.task = task
                    for split in splits: 
                        self.split = split
                        dir_path = f"outputs/{fw}/{model}/{task}/{split}/"
                        if os.path.isdir(dir_path):
                            self.evaluate_all_outputs(dir_path)

    def create_belief_instance_roc(self, out, eval):
        '''
        This method collects the intermediate predictions (beliefs) for the ROCStories task.

        @param
            out: model output
            eval: gold standard evaluation data
        @returns
            dictionary with the beliefs for each turn in the conversation
        '''
        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)
                
                # format as list not as dict
                if str in [type(item) for item in conv_list]:
                    pred = conv_list.copy()
                    conv_list = []

                    for pos, sent in enumerate(pred, start=1):
                        if sent != "":
                            conv_list.append({"sentence": sent, "position": pos})

                player = out["player"][i]

                if i == (len(out["turns"])-1) and conv_list == []:
                    pred = self.extract_answer_list(turn, eval["gold"])

                    for pos, sent in enumerate(pred, start=1):
                        if sent != "":
                            conv_list.append({"sentence": sent, "position": pos})

                if type(conv_list) == list:
                    # check for 0:
                    zero_count = True in [0 in conv_dir.values() for conv_dir in conv_list]

                    for conv_dir in conv_list:
                        if type(conv_dir) in [defaultdict, dict] and conv_dir["sentence"] in sent2gold.keys() and "position" in conv_dir.keys() and type(conv_dir["position"])==int:
                            if zero_count:
                                pos = conv_dir["position"]+1
                            else:
                                pos = conv_dir["position"]
                            
                            sent_idx = sent2gold[conv_dir["sentence"]]
                            
                            # Player, sentence, position, time-step
                            if sent_idx == pos:
                                correct = "correct"
                            else:
                                correct = "incorrect"

                            beliefs.append({
                                "sentence": conv_dir["sentence"],
                                "turn": i+1,
                                "agent": player,
                                "gold_sent_idx": sent_idx,
                                "hypo": int(conv_dir["position"]),
                                "correct": correct,
                                "label": f"({player}, sentence {sent_idx}, hypothesis {conv_dir['position']})"
                            })

        return {"beliefs": beliefs}
    
    def create_belief_instance_cmu(self, out, eval):
        '''
        This method collects the intermediate predictions (beliefs) for the CMU Booksum task.

        @param
            out: model output
            eval: gold standard evaluation data
        @returns
            dictionary with the beliefs for each turn in the conversation
        '''
        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)

                if i == (len(out["turns"])-1) and conv_list == []:
                    pred = self.extract_answer_list(turn, eval["gold"])
                    pred = self.extract_pred_odd(pred, out["titles"])

                    conv_list.append({"title": pred, "label": "true"})

                player = out["player"][i]

                if type(conv_list) == list:
                    for conv_dir in conv_list:
                        if type(conv_dir) in [defaultdict, dict] and conv_dir["title"] in out["titles"] and "label" in conv_dir.keys():                        
                            # Player, sentence, position, time-step
                            if "true" in conv_dir["label"].lower() and conv_dir["title"] == eval["gold"]:
                                correct = "correct"
                            elif "false" in conv_dir["label"].lower() and conv_dir["title"] != eval["gold"]:
                                correct = "correct"
                            else:
                                correct = "incorrect"

                            beliefs.append({
                                "title": conv_dir["title"],
                                "turn": i+1,
                                "agent": player,
                                "gold_title": eval["gold"],
                                "hypo": conv_dir["label"],
                                "correct": correct,
                                "label": f"({player}, title {conv_dir['title']}, hypothesis {conv_dir['label']})"
                            })

        return {"beliefs": beliefs}

    def evaluate_last_beliefs(self):
        '''
        This method evaluates the last hypothesis of the multi-agent outputs for both ROCStories and CMU Booksum tasks.

        @returns
            accuracy of the last hypothesis for each instance
        '''
        eval_file = f"{self.experiment_file.replace('outputs', 'eval').split('.jsonl')[0]}.jsonl"

        full_eval_data = self.load_eval_data(eval_file)

        if self.task == "rocstories":
            exp_beliefs = self.exp_data.map(lambda x, idx: self.create_belief_instance_roc(x, full_eval_data.select([idx])[0]), batched = False, with_indices=True)
            preds = exp_beliefs.map(self.create_pred_from_belief_roc, batched = False)
            self.eval_data = preds.map(lambda batch: self.calculate_metrics_roc(batch), batched = True)
        else:
            exp_beliefs = self.exp_data.map(lambda x, idx: self.create_belief_instance_cmu(x, full_eval_data.select([idx])[0]), batched = False, with_indices=True)
            preds = exp_beliefs.map(self.create_pred_from_belief_cmu, batched = False)
            self.eval_data = preds.map(lambda batch: self.calculate_metrics_cmu(batch), batched = True)
        
        self.save_eval_data(f"{eval_file.split('.jsonl')[0]}_last.jsonl")

        acc_mean = self.report_accuracy()
        return acc_mean


    def create_pred_from_belief_roc(self, beliefs):
        '''
        This method gets the final prediction from the beliefs for the ROCStories task.

        @param
            beliefs: dictionary with the beliefs for each turn in the conversation
        @returns
            dictionary with the final prediction for each instance
        '''
        # list of beliefs with dict, describing the belief
        pred = [""]*5

        for belief in beliefs["beliefs"][::-1]:
            if belief["agent"] not in ["Player 6", "expert"] and belief["sentence"] not in pred and int(belief["hypo"]) < len(pred):
                pred[int(belief["hypo"])-1] = belief["sentence"]

        return {"pred": pred}
    
    def create_pred_from_belief_cmu(self, beliefs):
        '''
        This method gets the final prediction from the beliefs for the CMU Booksum task.
        
        @param
            beliefs: dictionary with the beliefs for each turn in the conversation
        @returns
            dictionary with the final prediction for each instance'''
        # list of beliefs with dict, describing the belief
        pred = ""

        for belief in beliefs["beliefs"][::-1]:
            if belief["agent"] not in ["Player 6", "expert"] and belief["hypo"].lower() == "true":
                pred = belief["title"]

        return {"pred": pred}
    
    def calculate_metrics_roc(self, batch):
        '''
        This method calculates the evaluation metrics for the ROCStories task.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard, precision scores
        '''
        self.initiate_metric_lists()
        self.pred_lists = list(batch["pred"])
        for i in range(len(batch["storyid"])):

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

    def calculate_metrics_cmu(self, batch):
        '''
        This method calculates the evaluation metrics for the CMU Booksum task.

        @param
            batch: batch of instances
        @returns
            dictionary with the storyids, accuracy per instance, model prediction, gold standard
        '''
        self.initiate_metric_lists()
        self.pred_lists = list(batch["pred"])
        for i in range(len(batch["id"])):
            gold = batch["gold_title"][i]
            self.acc.append(self.compute_accuracy_odd(batch["pred"][i], gold, batch["titles"][i].index(gold)+1))
        
        return {"id": batch["id"], "accuracy": self.acc, "predictions": self.pred_lists, "gold": list(batch["gold_title"])}

    def load_eval_data(self, eval_file):
        '''
        This method loads the evaluation data from a JSON file.

        @param
            eval_file: path to the evaluation file
        @returns
            dataset: loaded evaluation dataset
        '''
        try:
            return load_dataset("json", data_files=eval_file, split="train")
        except:
            if self.task == "rocstories":
                self.evaluate_sent_order()
            else:
                self.evaluate_odd_one()
            return load_dataset("json", data_files=eval_file, split="train")

if __name__ == "__main__":
    outdir = "outputs/multi/Qwen3-32B/cmu_booksum/test/outputs_379"

    results = evaluate(outfile = outdir,framework= "multi", task = "cmu_booksum", structured=False, all = True, instances = 1000)
    print(results)

