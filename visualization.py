from datasets import load_dataset, load_from_disk
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import re
from pySankey.sankey import sankey
import numpy as np
from evaluate import Evaluator, evaluate
import plotly.graph_objects as go
from scipy.stats.mstats import ttest_ind
from scipy.stats import bootstrap
from matplotlib import colors
from matplotlib.patches import Patch
import math
from matplotlib.lines import Line2D

class Vis():
    def __init__(self, experiment_files: dict = {}, variation: bool = True, conv_format: str = "", last: bool = False, task: bool = "rocstories", instances: int = 1000):
        '''
        This class is used to visualize the results of the experiments. It can create bar plots for the mean accuracy of different setups, as well as Sankey diagrams to visualize the flow of beliefs across turns.

        @params
            experiment_files: A dictionary mapping experiment names to their output file paths.
            variation: A boolean indicating whether the ROCStories dataset with or without visualization is used.
            conv_format: The conversational format of the outputs.
            last: A boolean indicating whether to use only the last turn of the conversation as final answer.
            task: The task being evaluated (e.g., "rocstories").
        '''
        self.experiment_files = experiment_files

        self.last = last

        self.conv_format = conv_format

        self.task = task
        self.variation = variation
        self.evaluator = None

        if task == "rocstories":
            if self.variation == True:
                self.vispath = f"visualizations/{self.task}/variation/{conv_format}"
            elif self.variation == False:
                self.vispath = f"visualizations/{self.task}/no_variation/{conv_format}"
            else:
                self.vispath = f"visualizations/{self.task}/{conv_format}"
        else:
            self.vispath = f"visualizations/{self.task}/{conv_format}"

        if self.last:
            self.last_app = "_last"
        else:
            self.last_app = ""

        if experiment_files != {}:
            self.initialize_experiment_files(instances)

    def initialize_experiment_files(self, instances: int = 1000):
        '''
        This method loads the output datasets from disk and creates evaluation dataframes for each experiment setup.
        '''
        self.exp_data_hf = {k: load_from_disk(v).select(range(instances)) for k, v in self.experiment_files.items()}

        self.create_eval_dfs()

    def create_eval_dfs(self):
        '''
        This method creates evaluation dataframes for each experiment setup. It loads the evaluation data from the corresponding evaluation files and computes the mean predictions for each metric.
        '''
        if self.last:
            eval_files = defaultdict(str)
            for k, v in self.experiment_files.items():
                if "single" in k:
                    print("Setup is single, not performing last evaluation!")
                    eval_files[k] = f"{v.replace('outputs', 'eval')}.jsonl"
                else:
                    eval_files[k] = f"{v.replace('outputs', 'eval')}_last.jsonl"
        else:
            eval_files = {k: f"{v.replace('outputs', 'eval')}.jsonl" for k, v in self.experiment_files.items()}

        try:
            self.eval_data_hf = {k: self.load_eval_data(v, k) for k, v in eval_files.items()}
        except:
            self.eval_data_hf = {k: self.load_eval_data(v[:-6], k) for k, v in eval_files.items()}

        mean_preds = defaultdict(list)

        if self.task == "rocstories":
            self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]
        else:
            self.metrics = ["accuracy"]

        for setup, eval_data in self.eval_data_hf.items():
            for metric in self.metrics:
                mean_preds[setup].append(float(eval_data.with_format("pandas")[metric].mean()))

        self.mean_preds_df = pd.DataFrame(mean_preds.values(), index=mean_preds.keys(), columns=self.metrics) 

    def load_eval_data(self, eval_file, exp_name, instances: int = None):
        '''
        This method loads the evaluation data from the specified evaluation file. If the evaluation file is not found, it uses the Evaluator class to evaluate the experiment outputs and generate the evaluation data.

        @params
            eval_file: The path to the evaluation file.
            exp_name: The name of the experiment setup.

        @returns
            The evaluation data as a Hugging Face dataset.
        '''
        try:
            return load_dataset("json", data_files=eval_file, split="train")
        except:
            fw = ""
            if "single" in eval_file:
                fw = "single"
            elif "random" in eval_file:
                fw = "random"
            else:
                fw = "multi"

            evaluator = Evaluator(eval_file.replace("eval", "outputs"), fw, last = self.last, task = self.task, instances=instances)
            
            if self.last and fw == "multi":
                    evaluator.evaluate_last_beliefs()
            else:
                if self.task == "rocstories":
                    evaluator.evaluate_sent_order()
                else:
                    evaluator.evaluate_odd_one()
            
            return evaluator.eval_data

    def plot_performance(self):
        '''
        This method creates bar plots for the mean accuracy of different experiment setups. It saves the plots as PNG files in the specified visualization path.
        '''
        for metric in self.metrics:
            if self.variation:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: with variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 

                ax.bar_label(ax.containers[0])

                os.makedirs(self.vispath, exist_ok = True)
                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}.png")
            else:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: without variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 
                ax.bar_label(ax.containers[0])
                
                os.makedirs(self.vispath, exist_ok = True)
                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}.png")
            
            plt.close()

    def create_belief_instance_roc(self, index: int, setup: str):
        '''
        This method extracts the intermediate predictions for ROCStories.

        @params
            index: The index of the instance to extract.
            setup: The experiment setup name.
        @returns
            A list of dictionaries containing the beliefs for each turn, including the player, gold sentence index, hypothesis, correctness, and a label.
        '''
        out = self.exp_data_hf[setup].select([index])[0]
        eval = self.eval_data_hf[setup].select([index])[0]

        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        for i, turn in enumerate(out["turns"]):
            conv_list = self.create_conv_list_roc(turn)

            player = out["player"][i]

            for conv_dir in conv_list:
                if conv_dir["sentence"] in sent2gold.keys() and "position" in conv_dir.keys():
                    sent_idx = sent2gold[conv_dir["sentence"]]
                    
                    # Player, sentence, position, time-step
                    if sent_idx == conv_dir["position"]:
                        correct = "correct"
                    else:
                        correct = "incorrect"

                    beliefs.append({
                        "turn": i+1,
                        "agent": player,
                        "gold_sent": sent_idx,
                        "hypo": conv_dir["position"],
                        "correct": correct,
                        "label": f"({player}, sentence {sent_idx}, hypothesis {conv_dir['position']})"
                    })

        return beliefs
    
    def create_belief_instance_cmu(self, index: int, setup: str):
        '''
        This method extracts the intermediate predictions for CMU Booksum.

        @params
            index: The index of the instance to extract.
            setup: The experiment setup name.
        @returns
            A list of dictionaries containing the beliefs for each turn, including the player, gold sentence index, hypothesis, correctness, and a label.
        '''
        out = self.exp_data_hf[setup].select([index])[0]
        eval = self.eval_data_hf[setup].select([index])[0]

        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)

                player = out["player"][i]

                if type(conv_list) == list:
                    for conv_dir in conv_list:
                        if type(conv_dir) in [dict, defaultdict] and conv_dir["title"] in out["titles"] and "label" in conv_dir.keys():                        
                            # Player, sentence, position, time-step
                            if conv_dir["label"].lower() == "true" and conv_dir["title"] == eval["gold"]:
                                correct = "correct"
                            elif conv_dir["label"].lower() == "false" and conv_dir["title"] != eval["gold"]:
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

    def create_conv_list_roc(self, conv_str: str):
        '''
        This method extracts the predictions for ROCStories from the model output.

        @params
            conv_str: The conversation string from the model output.
        @returns
            A list of dictionaries containing the sentence, predicted position, reasoning, and arguments.
        '''
        conv_list = []
        sentences = [item[0] for item in re.findall(r"""sentence[\'"]: [\'"]([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        
        positions = re.findall(r"""position\': (\d)""", conv_str)
        reasoning_list = [item[0] for item in re.findall(r"""reasoning[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_1_list = [item[0] for item in re.findall(r"""argument_1[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_2_list = [item[0] for item in re.findall(r"""argument_2[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_3_list = [item[0] for item in re.findall(r"""argument_3[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_4_list = [item[0] for item in re.findall(r"""argument_4[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_5_list = [item[0] for item in re.findall(r"""argument_5[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]

        for i, sent in enumerate(sentences):
            conv_dict = defaultdict(str)

            conv_dict["sentence"] = sent

            if i < len(positions):
                conv_dict["position"] = int(positions[i])

            if i < len(reasoning_list):                        
                conv_dict["reasoning"] = self.strip_arg(reasoning_list[i])
            
            if i < len(reason_1_list):
                conv_dict["argument_1"] = self.strip_arg(reason_1_list[i])

            if i < len(reason_2_list):
                conv_dict["argument_2"] = self.strip_arg(reason_2_list[i])

            if i < len(reason_3_list):
                conv_dict["argument_3"] = self.strip_arg(reason_3_list[i])

            if i < len(reason_4_list):
                conv_dict["argument_4"] = self.strip_arg(reason_4_list[i])

            if i < len(reason_5_list):
                conv_dict["argument_5"] = self.strip_arg(reason_5_list[i])

            conv_list.append(conv_dict)
        
        return conv_list

    def create_conv_list_cmu(self, conv_str: str):
        '''
        This method extracts the predictions for CMU Booksum from the model output.

        @params
            conv_str: The conversation string from the model output.
        @returns
            A list of dictionaries containing the sentence, predicted position, reasoning, and arguments.
        '''
        conv_list = []

        titles = [item[0] for item in re.findall(r"""title[\'"]: [\'"]([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        
        labels = [item[0] for item in re.findall(r"""label[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
        reasoning_list = [item[0] for item in re.findall(r"""reasoning[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_1_list = [item[0] for item in re.findall(r"""argument_true[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]
        reason_2_list = [item[0] for item in re.findall(r"""argument_false[\'"]: [\'"]?([\w \.,-]+(\\'[\w \.,-]+)*)""", conv_str)]

        for i, title in enumerate(titles):
            conv_dict = defaultdict(str)

            conv_dict["title"] = title

            if i < len(labels):
                conv_dict["label"] = labels[i]

            if i < len(reasoning_list):                        
                conv_dict["reasoning"] = self.strip_arg(reasoning_list[i])
            
            if i < len(reason_1_list):
                conv_dict["argument_true"] = self.strip_arg(reason_1_list[i])

            if i < len(reason_2_list):
                conv_dict["argument_false"] = self.strip_arg(reason_2_list[i])
            
            conv_list.append(conv_dict)
        return conv_list
    
    def strip_arg(self, item):
        '''
        This method removes the prefixes "argument", "position", or "label" from the given item string.

        @params
            item: The string to be stripped.
        @returns
            The stripped string without the specified prefixes.
        '''
        if "argument" in item:
            item = item.strip("argument")
        elif "position" in item:
            item = item.strip("position")
        elif "label" in item:
            item = item.strip("label")
        return item
        
    def plot_sankey(self, setup: str):
        '''
        This method creates a Sankey diagram to visualize the flow of beliefs across turns for the specified experiment setup. It saves the diagram as a PNG file in the specified visualization path.

        @params
            setup: The experiment setup name.
        '''
        # last hypo == correct/incorrect -> final hypo == correct/incorrect 
        self.sankey_inst = []
        self.exp_data_hf[setup].map(lambda x, idx: self.create_sankey_hf(x, idx, setup), with_indices=True)

        df = pd.DataFrame(self.sankey_inst, columns = ["last_hypo", "final_hypo"])

        colors = {
            "correct": "#f78c1b",
            "incorrect": "#1b7ef7",
        }

        sankey(df["last_hypo"], df["final_hypo"], aspect=20, colorDict=colors, fontsize=12)

        fig = plt.gcf()

        os.makedirs(self.vispath, exist_ok = True)
        if self.variation:
            fig.savefig(f"{self.vispath}/sankey_{setup}_var.png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(f"{self.vispath}/sankey_{setup}_no_var.png", bbox_inches="tight", dpi=150)

    def create_sankey_hf(self, x, idx, setup):
        '''
        This method creates the data for the Sankey diagram by extracting the last and final hypotheses for each instance in the experiment setup.

        @params
            x: The instance data from the experiment setup.
            idx: The index of the instance.
            setup: The experiment setup name.
        '''
        if self.task == "rocstories":
            beliefs = self.create_belief_instance_roc(idx, setup)
        else:
            beliefs = self.create_belief_instance_cmu(idx, setup)

        seen_sents = []
        for belief in beliefs[::-1]:
            if belief["agent"] != "Player 6" and belief["gold_sent"] not in seen_sents:
                for final_sent_belief in beliefs[::-1]:
                    if final_sent_belief["agent"] == "Player 6" and final_sent_belief["gold_sent"] == belief["gold_sent"]:
                        self.sankey_inst.append([belief["correct"], final_sent_belief["correct"]])

    def plot_sankey_full_plotly(self, setup: str, sent_color: bool = False):
        '''
        This method creates a Sankey diagram using Plotly to visualize the flow of beliefs across turns for the specified experiment setup. It saves the diagram as an HTML file in the specified visualization path.

        @params
            setup: The experiment setup name.
            sent_color: A boolean indicating whether to color the links based on the sentence index.
        '''        
        self.num_turns = 0
        if "chain" in setup:
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif "star" in setup:
            self.num_turns = 2
        elif "graph" in setup.lower() or "layered" in setup.lower():
            self.num_turns = 3

        self.lbls = [f"{lbl}_{i}" for i in range(self.num_turns) for lbl in ["correct", "incorrect"]]
        colors = [color for i in range(self.num_turns) for color in ["#f78c1b", "#1b7ef7"]]

        self.links = defaultdict(int)
        self.sent_colors = {
            1: "#ffc7a2",
            2: "#e6adc0",
            3: "#93c4d5",
            4: "#bad0b9",
            5: "#cbb2ff"
        }

        self.correct_colors = {
            "correct": "#f78c1b", 
            "incorrect": "#1b7ef7"
        }

        self.exp_data_hf[setup].map(lambda x, idx: self.create_sankey_link(x, idx, setup, sent_color), with_indices=True)
        
        self.sources = [src for src, _, _ in self.links.keys()]
        self.targets = [tgt for _, tgt, _ in self.links.keys()]
        self.values = list(self.links.values())
        link_color = [col for _, _, col in self.links.keys()]

        percentage_lbls = []
        last_perc_b = 0.0
        for i in range(0, len(self.lbls)-2, 2):
            sum_a = sum([v for k, v in self.links.items() if k[0] == i])
            sum_b = sum([v for k, v in self.links.items() if k[0] == i+1])
            total_beliefs = sum_a + sum_b
            
            perc_a = round((sum_a/total_beliefs)*100, 2)
            perc_b = round((sum_b/total_beliefs)*100, 2)

            if i == 0:
                percentage_lbls.append(f"{self.lbls[i]}: {perc_a}%")
                percentage_lbls.append(f"{self.lbls[i+1]}: {perc_b}%")   
            else:
                percentage_lbls.append(f"{self.lbls[i]}: {perc_a}%")
                if perc_b > last_perc_b:
                    percentage_lbls.append(f"{self.lbls[i+1]}: +{round(perc_b - last_perc_b, 2)}")
                else:
                    percentage_lbls.append(f"{self.lbls[i+1]}: -{round(last_perc_b - perc_b, 2)}%")

            last_perc_b = perc_b

        perc_a = round(float(self.mean_preds_df.loc[setup]["accuracy"])*100, 2)
        perc_b = 100-perc_a
        percentage_lbls.append(f"{self.lbls[i]}: {perc_a}%")

        if perc_b > last_perc_b:
            percentage_lbls.append(f"{self.lbls[i+1].split('_')[0]}: +{round(perc_b - last_perc_b, 2)}")
        else:
            percentage_lbls.append(f"{self.lbls[i+1].split('_')[0]}: -{round(last_perc_b - perc_b, 2)}%")


        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = percentage_lbls,
                color = colors
            ),
            link = dict(
                source = self.sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
                target = self.targets,
                value = self.values,
                color = link_color
            ))])
        
        if self.variation:
            fig.update_layout(title_text=f"{setup} with variation: {self.conv_format}", font_size=10)
            fig.show()

            os.makedirs(self.vispath, exist_ok = True)
            if sent_color:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_var_sent.html")
            else:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_var.html")
        else:
            fig.update_layout(title_text=f"{setup} without variation: {self.conv_format}", font_size=10)
            fig.show()

            os.makedirs(self.vispath, exist_ok = True)
            if sent_color:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_no_var.html")
            else:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_no_var_sent.html")
        
    def create_sankey_link(self, inst, idx: int, setup: str, sent_color: bool = False):
        '''
        This method creates the links for the Sankey diagram.

        @params
            inst: The instance data from the experiment setup.
            idx: The index of the instance.
            setup: The experiment setup name.
            sent_color: A boolean indicating whether to color the links based on the sentence index.
        '''
        beliefs = self.create_belief_instance_roc(idx, setup)

        for sent_idx in range(1, 6):
            sent_beliefs = [belief for belief in beliefs if belief["gold_sent"] == sent_idx]
            turn_idxs = [(belief_idx, self.get_turn_idx(setup, belief)) for belief_idx, belief in enumerate(sent_beliefs)]

            for belief_idx, turn_idx in turn_idxs:
                curr_belief = sent_beliefs[belief_idx]

                if turn_idx <= turn_idxs[-1][1]:
                    next_idx = [next_item[1] for next_item in turn_idxs if next_item[1] > turn_idx][0]

                    next_targets = [item for item in turn_idxs if item[1] == next_idx]

                    # start sources for next beliefs
                    for target_idx, target_turn in next_targets:
                        source_lbl = f"{curr_belief['correct']}_{turn_idx}"

                        target_belief = sent_beliefs[target_idx]
                        target_lbl = f"{target_belief['correct']}_{target_turn}"

                        if sent_color:
                            self.links[(self.lbls.index(source_lbl), self.lbls.index(target_lbl), self.sent_colors[sent_idx])] += 1
                        else: 
                            self.links[(self.lbls.index(source_lbl), self.lbls.index(target_lbl), self.correct_colors[curr_belief["correct"]])] += 1

    def get_target(self, belief_idx, curr_turn_idx, beliefs, setup):
        '''
        This method retrieves the target label for the Sankey diagram.

        @params
            belief_idx: The index of the current belief.
            curr_turn_idx: The turn index of the current belief.
            beliefs: The list of beliefs for the instance.
            setup: The experiment setup name.
        '''
        curr_belief = beliefs[belief_idx]

        next_belief = None
        idx = belief_idx + 1
        while idx < len(beliefs):
            next_belief = beliefs[idx]
            next_turn_idx = self.get_turn_idx(setup, next_belief)

            if next_belief["gold_sent"] == curr_belief["gold_sent"] and next_turn_idx > curr_turn_idx:
                target_lbl = f"{next_belief['correct']}_{next_turn_idx}"
                self.targets.append(self.lbls.index(target_lbl)) 

                idx = len(beliefs)
            else:
                idx += 1

    def get_turn_idx(self, setup, belief):
        '''
        This method retrieves the turn index for the given belief based on the experiment setup.

        @params
            setup: The experiment setup name.
            belief: The belief dictionary containing the turn information.
        @returns
            The turn index for the belief.
        '''
        turn_idx = 0
        if "chain" in setup:
            turn_idx = belief["turn"]-1
        elif "star" in setup:
            if belief["agent"] != "Player 6":
                turn_idx = 0
            else:
                turn_idx = 1
        elif "graph" in setup.lower() or "layered" in setup.lower():
            if belief["turn"] < 6:
                turn_idx = 0
            elif belief["turn"] > 6 and belief["agent"] != "Player 6":
                turn_idx = 1
            else:
                turn_idx = 2
        
        return turn_idx
    
    def perform_ttest(self, comparison = "single", exp_name = ""):
        '''
        This method performs a Welch's t-test to compare the accuracy of the specified comparison setup against all other setups. It saves the results as a CSV file in the specified visualization path.

        @params
            comparison: The experiment setup that is tested to be significantly lower than the other setups.
            exp_name: An optional name for the experiment to include in the output file name.
        '''
        first_accs = [list(eval_data['accuracy']) for setup, eval_data in self.eval_data_hf.items() if comparison in setup][0]

        stats_dict = defaultdict(list)
        for setup, eval_data in self.eval_data_hf.items():
            if comparison not in setup:
                second_accs = list(eval_data['accuracy'])
                
                stats = ttest_ind(first_accs, second_accs, equal_var=False, alternative="less")

                stats_dict["first_exp_file"].append(self.experiment_files[comparison])
                stats_dict["first_setup"].append(comparison)
                stats_dict["first_mean"].append(np.mean(first_accs))

                stats_dict["second_exp_file"].append(self.experiment_files[setup])
                stats_dict["second_setup"].append(setup)
                stats_dict["second_mean"].append(np.mean(second_accs))
                stats_dict["Welch's t-test"].append(stats.statistic)
                stats_dict["p-value"].append(stats.pvalue)

                if stats.pvalue < 0.05:
                    stats_dict["significant"] = True
                else:
                    stats_dict["significant"] = False

        stats_df = pd.DataFrame(stats_dict)

        os.makedirs(self.vispath, exist_ok = True)
        outfile = f"{self.vispath}significance_{comparison}"
        if exp_name != "":
            outfile += f"_{exp_name}"

        if self.last:
            outfile += f"_last"

        outfile += ".csv"
        stats_df.to_csv(outfile, sep = ";")

    def perform_paired_bootstrap(self):
        '''
        This method performs a paired bootstrap test to compare the accuracy of the specified comparison setup against all other setups. It saves the results as a CSV file in the specified visualization path.

        @params
            comparison: The experiment setup that is tested to be significantly lower than the other setups.
            exp_name: An optional name for the experiment to include in the output file name.
        '''
        self.last = False
        stats_dict = defaultdict(list)

        # 1. load all experiment files
        run_file = "eval/final.csv"

        df_experiment_files = pd.read_csv(run_file, delimiter = ";")

        tasks = ["StoryCon", "BookDiff"]
        models = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "Qwen3-32B"]
        exp_comparisons = {
            "Single": ["Base", "Improved", "Hypo", "Full", "Args", "Reject", "Support", "Correct", "Incorrect"],
            "Hypo": ["Args"],
            "Support": ["Reject"],
            "Incorrect": ["Correct"]
        }

        # hyperparameters
        m = len(tasks) * len(models) * sum([len(higher_exps) for higher_exps in exp_comparisons.values()]) # number of comparisons for bonferroni correction
        alpha = 0.05
        

        # 2. bootstrapping
        for task in tasks:
            if task == "StoryCon":
                self.task = "rocstories"
            else:
                self.task = "cmu_booksum"

            for model in models:
                for lower_exp, higher_exps in exp_comparisons.items():

                    # 2.1 get scores for lower comparisons 
                    # get exp files for all three runs
                    lower_exp_row = df_experiment_files.loc[lambda df_experiment_files: df_experiment_files["task"] == task].loc[df_experiment_files["model"] == model].loc[df_experiment_files["framework"] == lower_exp]

                    # get eval data for all three runs
                    lower_eval_0 = self.load_eval_data(lower_exp_row["output_0"].item().replace("outputs", "eval"), lower_exp.lower(), instances=1000)
                    lower_eval_1 = self.load_eval_data(lower_exp_row["output_1"].item().replace("outputs", "eval"), lower_exp.lower(), instances=1000)
                    lower_eval_2 = self.load_eval_data(lower_exp_row["output_2"].item().replace("outputs", "eval"), lower_exp.lower(), instances=1000)

                    # compute mean per instance across runs
                    scores_A = pd.DataFrame([lower_eval_0["accuracy"], lower_eval_1["accuracy"], lower_eval_2["accuracy"]]).mean(axis=0)

                    # 2.2 get higher comparison
                    for higher_exp in higher_exps:
                        stats_dict["task"].append(task)
                        stats_dict["model"].append(model)
                        stats_dict["first_setup"].append(lower_exp)
                        stats_dict["first_exp_files"].append(", ".join([lower_exp_row[f"output_{i}"].item() for i in range(3)]))

                        # get exp files for all three runs
                        higher_exp_row = df_experiment_files.loc[lambda df_experiment_files: df_experiment_files["task"] == task].loc[df_experiment_files["model"] == model].loc[df_experiment_files["framework"] == higher_exp]
                        stats_dict["second_setup"].append(higher_exp)

                        # get eval data for all three runs
                        higher_eval_0 = self.load_eval_data(higher_exp_row["output_0"].item().replace("outputs", "eval"), higher_exp.lower(), instances=1000)
                        higher_eval_1 = self.load_eval_data(higher_exp_row["output_1"].item().replace("outputs", "eval"), higher_exp.lower(), instances=1000)
                        higher_eval_2 = self.load_eval_data(higher_exp_row["output_2"].item().replace("outputs", "eval"), higher_exp.lower(), instances=1000)

                        stats_dict["second_exp_files"].append(", ".join([higher_exp_row[f"output_{i}"].item() for i in range(3)]))

                        # compute mean per instance across runs
                        scores_B = pd.DataFrame([higher_eval_0["accuracy"], higher_eval_1["accuracy"], higher_eval_2["accuracy"]]).mean(axis=0)

                        # 2.3 compute difference between compared runs
                        d = scores_A - scores_B

                        # 2.4 perform bootstrap
                        res = bootstrap((d, ), np.mean, n_resamples=100000, confidence_level=1-alpha/m, method = "percentile", alternative = "less", rng = 0)
                        stats_dict["corrected_alpha"].append(alpha/m)

                        # 2.5 compute p-value, significance, confidence interval
                        low, high = res.confidence_interval
                        stats_dict["confidence_interval"].append(tuple((float(res.confidence_interval[0]), float(res.confidence_interval[1])))) 
                        stats_dict["significant"].append(bool(high < 0)) 
                        stats_dict["standard_error"].append(float(res.standard_error))

                        null = res.bootstrap_distribution - d.mean()
                        stats_dict["p-value"].append(float((np.sum(null <= d.mean()) + 1) / (len(null) + 1)))

        stats_df = pd.DataFrame(stats_dict)

        outfile = f"./visualizations/paired_bootstrap.csv"
        stats_df.to_csv(outfile, sep = ";")

    def correlate_last_final(self, setup):
        '''
        This method correlates the accuracy of the last turn predictions with the final performance for the specified experiment setup. It creates a scatter plot to visualize the correlation and saves it as a PNG file in the specified visualization path.
        @params
            setup: The experiment setup name.
        '''
        if self.last:
            mean_last = self.mean_preds_df
            
            self.last = False
            self.create_eval_dfs()
            mean_final = self.mean_preds_df
        else:
            mean_final = self.mean_preds_df
            
            self.last = True
            self.create_eval_dfs()
            mean_last = self.mean_preds_df

        label = [mean_final.index[i].split("_")[1] for i in range(len(mean_final.index))]
        df = pd.concat([mean_final["accuracy"].rename("final"), mean_last["accuracy"].rename("last")], axis = 1)

        print(df.corr())
        colorlist = list(colors.ColorConverter.colors.keys())
        fig, ax = plt.subplots()
        [df.iloc[[i]].plot.scatter('last', 'final', ax=ax, s=50, label=l,
                                color=colorlist[i % len(colorlist)])
        for i,l in enumerate(label)]
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(f"Correlation between MAS and final performance: {setup}")

        os.makedirs(self.vispath, exist_ok = True)
        plt.savefig(f"{self.vispath}/last_final_correlation_{setup}.png")
        plt.close()

    def plot_all_table(self, run_file = "eval/args_hypos_runs_llama_roc.csv"):
        """
        This function plots the accuracy and accuracy changes of the setups in run_file.

        @params
            run_file: The path to the CSV file containing the experiment setups and their corresponding output files
        """
        df_experiment_files = pd.read_csv(run_file, delimiter = ";").set_index("format")

        df_acc = df_experiment_files["run"].map(self.get_avg_metric)

        # plot sankeys
        if self.last == False:
            if run_file == "eval/multi/Qwen3-32B/rocstories/dev/full_full_topologies_250.csv":
                df_dict = df_experiment_files.to_dict()
            else:
                df_dict = df_experiment_files.T.to_dict()
            acc_changes = defaultdict(list)
            for conv_format, exp_files in df_dict.items():
                if conv_format == "run":
                    conv_format = "full_full"
                
                self.__init__(experiment_files = exp_files, variation = self.variation, conv_format = conv_format, last = self.last)
                
                for setup in exp_files.keys():
                    if "single" not in setup and "random" not in setup and "du" not in setup:
                        if conv_format != "run":
                            acc_changes[f"{setup}_{conv_format}"] = self.get_perf_changes(setup)
                        else:
                            acc_changes[f"{setup}"] = self.get_perf_changes(setup)
                    else:
                        acc_changes[f"{setup}"] = [float(df_acc[setup][conv_format])]*11

            df_acc_changes = pd.DataFrame(acc_changes)
            df_acc_changes.plot(ylabel="Micro precision", xlabel="Turn", xticks = np.arange(11), yticks=np.arange(0, 101, 10))

            os.makedirs(self.vispath, exist_ok = True)
            plt.savefig(f"{self.vispath}/acc_changes.png")
        
        df_acc.to_csv(f"./visualizations/{self.task}/all{self.last_app}.csv", sep = ";")

    def plot_acc_changes(self, run_file: str = None, df_experiment_files = None, instances: int = 1000, one_model: bool = False):
        '''
        This method plots the accuracy changes of the setups in run_file or df_experiment_files.

        @params
            run_file: The path to the CSV file containing the experiment setups and their corresponding output files
            df_experiment_files: A DataFrame containing the experiment setups and their corresponding output files
        '''
        if run_file:
            df_experiment_files = pd.read_csv(run_file, delimiter = ";")
            df_experiment_files["accuracy"] = df_experiment_files["output"].map(lambda x: self.get_avg_metric(x, instances=instances))

        # plot sankeys
        df_dict = df_experiment_files.T.to_dict()
        acc_changes = defaultdict(list)
        for row in df_dict.values():
            conv_format = row["format"]

            if row["task"].lower() == "storycon":
                task = "rocstories"
            elif row["task"].lower() == "bookdiff":
                task = "cmu_booksum"
            else:
                task = row["task"]

            if "llama" in row["model"].lower():
                model = "llama"
            elif row["model"].lower() == "qwen3-8b":
                model = "qwen"
            else:
                model = "qwen32"

            exp_file = row["output"]

            if one_model:
                setup = conv_format
            else:
                setup = f"{conv_format}_{model}"

            if conv_format.lower() not in ["single", "random", "base", "improved"]:
                self.__init__(experiment_files = {setup: exp_file}, variation = self.variation, conv_format = conv_format, last = self.last, task = task.lower(), instances = instances)
                
                self.evaluator = Evaluator(exp_file,"multi", task.lower())

                if self.task == "rocstories":
                    self.num_turns = 11
                else:
                    self.num_turns = 7

                self.corr_per_turn = [0]*self.num_turns
                self.beliefs_per_turn = [0]*self.num_turns
                self.exp_data_hf[setup] = self.exp_data_hf[setup].select(range(instances)).map(lambda x, idx: self.get_acc_per_turn(x, idx, setup), with_indices=True)

                acc_per_turn = [0]*self.num_turns
                for i in range(self.num_turns):
                    if self.beliefs_per_turn[i] != 0:
                        acc_per_turn[i] = round((self.corr_per_turn[i]/self.beliefs_per_turn[i])*100, 2)
                
                acc_changes[setup] = acc_per_turn

        df_acc_changes = pd.DataFrame(acc_changes)

        return df_acc_changes

    def plot_arg_hypo_changes(self):
        '''
        This method plots Fig. 4a, i.e., the accuracy changes of the argument and hypothesis setups for both ROCStories and CMU Booksum tasks.
        '''
        self.task = "rocstories"
        df_experiment_files = pd.read_csv("eval/args_hypo_runs_roc.csv", delimiter = ";")
        #df_experiment_files = pd.read_csv("eval/debug.csv", delimiter = ";")

        roc_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            roc_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        roc = pd.DataFrame(columns=roc_dfs[0].columns)
        err_roc = pd.DataFrame(columns=roc_dfs[0].columns)
        # , index=["Hypo_llama", "Full_llama", "Args_llama", "Hypo_qwen", "Full_qwen", "Args_qwen"]
        for col in roc_dfs[0].columns:
            roc_col_df = pd.DataFrame([roc_dfs[0][col], roc_dfs[1][col], roc_dfs[2][col]])
            #roc[col] = (roc_dfs[0][col] + roc_dfs[1][col] + roc_dfs[2][col])/3
            roc[col] = roc_col_df.mean()
            err_roc[col] = roc_col_df.std()
    
        self.task = "cmu_booksum"
        df_experiment_files = pd.read_csv("eval/args_hypo_runs_cmu.csv", delimiter = ";")

        cmu_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            cmu_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        cmu = pd.DataFrame(columns=cmu_dfs[0].columns)
        err_cmu = pd.DataFrame(columns=cmu_dfs[0].columns)
        for col in cmu_dfs[0].columns:
            #cmu[col] = (cmu_dfs[0][col] + cmu_dfs[1][col] + cmu_dfs[2][col])/3

            cmu_col_df = pd.DataFrame([cmu_dfs[0][col], cmu_dfs[1][col], cmu_dfs[2][col]])

            cmu[col] = cmu_col_df.mean()
            err_cmu[col] = cmu_col_df.std()

        df_list = [cmu, roc]
        err_df_list = [err_cmu, err_roc]
        
        nrow = 1
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Hypo_llama': "#F08F2D",
            'Full_llama': "#3C63E2",
            'Args_llama': '#BE77E4',
            'Hypo_qwen': "#ffc7a2",
            'Full_qwen': "#17b2e1",
            'Args_qwen': "#cbb2ff",
            'Hypo_qwen32': "#F17C08",
            'Full_qwen32': "#0339EC",                
            'Args_qwen32': "#A824EE",
        }

        linestyle = {
            'Hypo_llama': 'dashed',# dashdot
            'Full_llama': "dashed",
            'Args_llama': 'dashed',
            'Hypo_qwen': '-',
            'Full_qwen': '-',
            'Args_qwen': '-',
            'Hypo_qwen32': 'dotted',
            'Full_qwen32': 'dotted',
            'Args_qwen32': 'dotted'
        }

        # marker_map = {
        #     'Hypo_llama': '^',
        #     'Hypo_qwen':  's',
        #     'Full_llama': '^',
        #     'Full_qwen':  's',
        #     'Args_llama': '^',
        #     'Args_qwen':  's',
        # }
        subtitle = ["BookDiff", "StoryCon"]

        for c in range(ncol):
            if c == 0:
                ylabel = "Micro Precision"
                xticks = np.arange(7)
                yticks=np.arange(0, 81, 10)
                linepos = 2
                finalpos = 5
            else:
                ylabel = ""
                xticks = np.arange(11)
                linepos = 4
                finalpos = 9,
                #yticks = None

            df_list[c].plot(
                ax=axes[c], 
                ylabel=ylabel, 
                xlabel=f"Turn in {subtitle[c]}", 
                xticks = xticks, 
                yticks=yticks, 
                color=color_map,
                #title = subtitle[c], 
                legend = False,
                #figsize=(4.5,2.5),
                figsize=(4,2.25), #wxh
                #marker='o',
                #markersize=3
                zorder= 2,
                yerr=err_df_list[c],
                #capsize = 2
            )
            axes[c].axvline(linepos, color='lightgray', linestyle='--', zorder=1)
            axes[c].axvline(finalpos, color='darkgray', linestyle='--', zorder=1)

            # make Qwen3-8B dashed
            for line in axes[c].get_lines():
                col_name = line.get_label()
                if col_name in linestyle:
                    line.set_linestyle(linestyle[col_name])
                    #line.set_marker(marker_map.get(col_name, 'o'))
                    #line.set_markersize(4)
                    #line.set_markeredgecolor('black')  
                    #line.set_markeredgewidth(0.1)

            # if c == 0:
            #     axes[c].legend(loc = "upper left")

            if c == 0:
                # First legend: Colors (upper left)
                color_legend = [
                    Line2D([0], [0], color='#cbb2ff', lw=2, linestyle='-', label='Args'),
                    Line2D([0], [0], color='#ffc7a2', lw=2, linestyle='-', label='Hypo'),
                    Line2D([0], [0], color='#17b2e1', lw=2, linestyle='-', label='Full'),
                ]
                legend1 = axes[c].legend(
                    handles=color_legend, 
                    loc="lower left", 
                    fontsize='small',
                    handletextpad=0.4,     # Less space between line and text (default is 0.8)
                    borderpad=0.3,         # Less padding inside the box (default is 0.4)
                    borderaxespad=0.2,     # Less space between legend and axes edge
                )
                axes[c].add_artist(legend1)  # <-- This keeps the first legend!
                
                # Second legend: Line styles (upper right)
                style_legend = [
                    Line2D([0], [0], color='darkgray', lw=2, linestyle=(0, (2.5, 0.5, 2.5, 0.5)), label='Llama'),
                    Line2D([0], [0], color='lightgray', lw=2, linestyle='-', label='Qwen-8B'),
                    Line2D([0], [0], color='dimgray', lw=2, linestyle='dotted', label='Qwen-32B')
                ]
                axes[c].legend(
                    handles=style_legend, 
                    loc="lower right", 
                    fontsize='small',
                    handlelength=1.0,      # Shorter line sample (default is 2.0)
                    handletextpad=0.4,     # Less space between line and text (default is 0.8)
                    borderpad=0.3,         # Less padding inside the box (default is 0.4)
                    borderaxespad=0.2,     # Less space between legend and axes edge
                )

            if c == 1:
                axes[c].tick_params(labelleft=False)    
        
        plt.tight_layout() 
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"visualizations/args_hypos.pdf")

    def plot_exploration_changes(self):
        '''
        This method plots Fig. 4b, i.e., the accuracy changes of the reject and support for both ROCStories and CMU Booksum tasks.
        '''
        self.task = "rocstories"
        df_experiment_files = pd.read_csv("eval/exploration_runs_roc.csv", delimiter = ";")

        roc_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            roc_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        roc = pd.DataFrame(columns=roc_dfs[0].columns)
        for col in roc_dfs[0].columns:
            roc[col] = (roc_dfs[0][col] + roc_dfs[1][col] + roc_dfs[2][col])/3

        self.task = "cmu_booksum"
        df_experiment_files = pd.read_csv("eval/exploration_runs_cmu.csv", delimiter = ";")

        cmu_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            cmu_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        cmu = pd.DataFrame(columns=cmu_dfs[0].columns)
        for col in cmu_dfs[0].columns:
            cmu[col] = (cmu_dfs[0][col] + cmu_dfs[1][col] + cmu_dfs[2][col])/3

        df_list = [cmu, roc]
        
        nrow = 1
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Reject_llama': '#BE77E4',
            'Support_llama': '#F08F2D',  
            'Reject_qwen': '#cbb2ff',
            'Support_qwen': '#ffc7a2',
            'Reject_qwen32': "#A824EE",
            'Support_qwen32': "#F17C08",                        
        }

        linestyle = {
            'Reject_llama': 'dashed',
            'Support_llama': "dashed",
            'Reject_qwen': '-',
            'Support_qwen': '-',
            'Reject_qwen32': 'dotted',
            'Support_qwen32': 'dotted',
        }

        # marker_map = {
        #     'Reject_llama': '^',
        #     'Support_llama':  '^',
        #     'Reject_qwen': 's',
        #     'Support_qwen':  's',
        # }
        subtitle = ["BookDiff", "StoryCon"]

        for c in range(ncol):
            if c == 0:
                ylabel = "Micro Precision"
                yticks=np.arange(0, 81, 10)
                xticks = np.arange(7)
                linepos = 2
                finalpos = 5
            else:
                ylabel = ""
                xticks = np.arange(11)
                linepos = 4
                finalpos = 9

            df_list[c].plot(
                ax=axes[c], 
                ylabel=ylabel, 
                xlabel=f"Turn in {subtitle[c]}", 
                xticks = xticks, 
                yticks=yticks, 
                color=color_map, 
                legend = False,
                figsize=(4,2.25),
                zorder=2
            )
            axes[c].axvline(linepos, color='lightgray', linestyle='--', zorder=1)
            axes[c].axvline(finalpos, color='darkgray', linestyle='--', zorder=1)

            # make Qwen3-8B dashed
            for line in axes[c].get_lines():
                col_name = line.get_label()
                if col_name in linestyle:
                    line.set_linestyle(linestyle[col_name])
                    #line.set_marker(marker_map[col_name])
                    #line.set_markersize(3)
                    #line.set_markeredgecolor('black')  
                    #line.set_markeredgewidth(0.1)

            # if c == 0:
            #     axes[c].legend(loc = "upper left")

            if c == 0:
                # First legend: Colors (upper left)
                color_legend = [
                    Line2D([0], [0], color='#cbb2ff', lw=2, linestyle='-', label='Reject'),
                    Line2D([0], [0], color='#ffc7a2', lw=2, linestyle='-', label='Support'),
                ]
                legend1 = axes[c].legend(
                    handles=color_legend, 
                    loc="lower left", 
                    fontsize='small',
                    handletextpad=0.4,     # Less space between line and text (default is 0.8)
                    borderpad=0.3,         # Less padding inside the box (default is 0.4)
                    borderaxespad=0.2,     # Less space between legend and axes edge
                )
                axes[c].add_artist(legend1)  # <-- This keeps the first legend!

            if c == 1:
                axes[c].tick_params(labelleft=False)    
        
        plt.tight_layout() 
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"visualizations/exploration.pdf")

    def plot_correctness_changes(self):
        '''
        This method plots Fig. 4c, i.e., the accuracy changes of correct and incorrect for both ROCStories and CMU Booksum tasks.
        '''
        self.task = "rocstories"
        df_experiment_files = pd.read_csv("eval/correctness_runs_roc.csv", delimiter = ";")

        roc_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            roc_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        roc = pd.DataFrame(columns=roc_dfs[0].columns)
        for col in roc_dfs[0].columns:
            roc[col] = (roc_dfs[0][col] + roc_dfs[1][col] + roc_dfs[2][col])/3

        self.task = "cmu_booksum"
        df_experiment_files = pd.read_csv("eval/correctness_runs_cmu.csv", delimiter = ";")

        cmu_dfs = [] 
        version_ids = [0, 1, 2]
        for i in version_ids:
            drop = [idx for idx in version_ids if idx != i]

            df = df_experiment_files
            for drop_idx in drop:
                df = df.drop(f"output_{drop_idx}", axis = 1) # specify axis to remove columns

            df = df.rename(columns={f"output_{i}":"output"})

            cmu_dfs.append(self.plot_acc_changes(df_experiment_files = df))

        cmu = pd.DataFrame(columns=cmu_dfs[0].columns)
        for col in cmu_dfs[0].columns:
            cmu[col] = (cmu_dfs[0][col] + cmu_dfs[1][col] + cmu_dfs[2][col])/3

        df_list = [cmu, roc]
        
        nrow = 1
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Correct_llama': '#BE77E4',
            'Incorrect_llama': "#EA9D50",
            'Correct_qwen': '#cbb2ff',
            'Incorrect_qwen': '#ffc7a2',       
            'Correct_qwen32': "#A824EE",
            'Incorrect_qwen32': "#F17C08",                
        }

        linestyle = {
            'Correct_llama': 'dashed',
            'Incorrect_llama': "dashed",
            'Correct_qwen': '-',
            'Incorrect_qwen': '-',
            'Correct_qwen32': 'dotted',
            'Incorrect_qwen32': 'dotted',
        }

        # marker_map = {
        #     'Correct_llama': '^',
        #     'Incorrect_llama':  '^',
        #     'Correct_qwen': 's',
        #     'Incorrect_qwen':  's',
        # }
        subtitle = ["BookDiff", "StoryCon"]

        for c in range(ncol):
            if c == 0:
                ylabel = "Micro Precision"
                xticks = np.arange(7)
                linepos = 2
                finalpos = 5
                yticks=np.arange(0, 81, 10)
            else:
                ylabel = ""
                xticks = np.arange(11)
                linepos = 4
                finalpos = 9

            df_list[c].plot(
                ax=axes[c], 
                ylabel=ylabel, 
                xlabel=f"Turn in {subtitle[c]}",
                xticks = xticks, 
                yticks=yticks, 
                color=color_map,  
                legend = False,
                figsize=(4,2.25),
                zorder = 2
            )
            axes[c].axvline(linepos, color='lightgray', linestyle='--', zorder=1)
            axes[c].axvline(finalpos, color='darkgray', linestyle='--', zorder=1)
            
            # make Qwen3-8B dashed
            for line in axes[c].get_lines():
                col_name = line.get_label()
                if col_name in linestyle:
                    line.set_linestyle(linestyle[col_name])
                    #line.set_marker(marker_map[col_name])
                    #line.set_markersize(3)
                    #line.set_markeredgecolor('black')  
                    #line.set_markeredgewidth(0.1)

            # if c == 0:
            #     axes[c].legend(loc = "upper left")

            if c == 0:
                # First legend: Colors (upper left)
                color_legend = [
                    Line2D([0], [0], color='#cbb2ff', lw=2, linestyle='-', label='Correct'),
                    Line2D([0], [0], color='#ffc7a2', lw=2, linestyle='-', label='Incorrect'),
                ]
                legend1 = axes[c].legend(
                    handles=color_legend, 
                    loc="lower left", 
                    fontsize='small',
                    handletextpad=0.4,     # Less space between line and text (default is 0.8)
                    borderpad=0.3,         # Less padding inside the box (default is 0.4)
                    borderaxespad=0.2,     # Less space between legend and axes edge
                )
                axes[c].add_artist(legend1)  # <-- This keeps the first legend!

            if c == 1:
                axes[c].tick_params(labelleft=False)    
        
        plt.tight_layout() 
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"visualizations/correctness.pdf")

    def get_acc_per_turn(self, inst, idx, setup):
        '''
        This method calculates the accuracy per turn for the specified instance in the experiment setup.

        @params
            inst: The instance data from the experiment setup.
            idx: The index of the instance.
            setup: The experiment setup name.
        '''
        out = self.exp_data_hf[setup].select([idx])[0]
        eval = self.eval_data_hf[setup].select([idx])[0]

        if self.task == "rocstories":
            beliefs = self.evaluator.create_belief_instance_roc(out, eval)
        else:
            beliefs = self.evaluator.create_belief_instance_cmu(out, eval)

        for belief in beliefs["beliefs"]:
            if belief["correct"] == "correct":
                self.corr_per_turn[belief["turn"]-1] += 1

            self.beliefs_per_turn[belief["turn"]-1] += 1
        
    def get_avg_metric(self, exp_file: str, metric: str = "accuracy", round_acc: bool = True, instances: int = 1000):
        '''
        This method calculates the average accuracy for the specified experiment file.

        @params
            exp_file: The path to the experiment output file.
            metric: The metric to calculate the average for (default is "accuracy").
        @returns
            The average value of the specified metric over all instances.
        '''
        if "booksum" in exp_file:
            self.task = "cmu_booksum"
        else:
            self.task = "rocstories"

        if "multi" in exp_file:
            framework = "multi"
        elif "single" in exp_file:
            framework = "single"
        else:
            framework = "random"

        if self.last and framework != "single":
            results = evaluate(exp_file, framework = framework, task = self.task, structured = True, all=True, instances=instances)
            avg_metric = results[f"{metric}_last"]
        elif framework == "random":
            eval_data = load_dataset("json", data_files = f"{exp_file.replace("outputs", "eval")}.jsonl", split = "train")
            avg_metric = sum(eval_data[metric])/len(eval_data)
        else:
            results = evaluate(exp_file, framework = framework, task = self.task, structured = False, all=True, instances=instances)
            avg_metric = results[metric]

        if round_acc:
            return round(avg_metric*100, 2)
        else:
            return avg_metric
        
    def get_avg_metrics(self, exp_file: str, instances: int = 1000, inlcude_random: bool = False):
        '''
        This method calculates the average accuracy for the specified experiment file.

        @params
            exp_file: The path to the experiment output file.
            metric: The metric to calculate the average for (default is "accuracy").
        @returns
            The average value of the specified metric over all instances.
        '''
        if "booksum" in exp_file:
            self.task = "cmu_booksum"
        else:
            self.task = "rocstories"

        if "multi" in exp_file:
            framework = "multi"
        elif "single" in exp_file:
            framework = "single"
        else:
            framework = "random"

        if self.last and framework != "single":
            return evaluate(exp_file, framework = framework, task = self.task, structured = True, all=True, instances=instances)
        else:
            return evaluate(exp_file, framework = framework, task = self.task, structured = False, all=True, instances=instances)
    
    def get_sanity_checks(self, exp_file):
        '''
        This method performs sanity checks on the specified experiment file.

        @params
            exp_file: The path to the experiment output file.
        @returns
            fraction of instances returned in the wrong format
        '''
        if "booksum" in exp_file:
            self.task = "cmu_booksum"
        else:
            self.task = "rocstories"

        if self.last and "single" not in exp_file:
            eval_file = f"{exp_file.replace('outputs', 'eval')}_last.jsonl"
        else:
            eval_file = f"{exp_file.replace('outputs', 'eval')}.jsonl"
        
        if self.last and "single" not in exp_file:
            wrong_format, wrong_format_last = evaluate(exp_file, framework = eval_file.split("/")[1], task = self.task, structured = True, sanity_checks=True)
            wf = wrong_format_last
        else:
            wrong_format, wrong_format_last = evaluate(exp_file, framework = eval_file.split("/")[1], task = self.task, structured = False, sanity_checks=True)
            wf = wrong_format

        return round(wf, 3)
    
    def get_perf_changes(self, setup, sent_color = False):
        '''
        This method computes the intermediate precision per turn.

        @returns 
            acc_per_turn: micro precision per turn
        '''
        self.num_turns = 0
        if "chain" in setup:
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif "star" in setup.lower():
            self.num_turns = 2
        elif "graph" in setup.lower() or "layered" in setup.lower():
            self.num_turns = 3

        self.lbls = [f"{lbl}_{i}" for i in range(self.num_turns) for lbl in ["correct", "incorrect"]]

        self.links = defaultdict(int)
        self.sent_colors = {
            1: "#ffc7a2",
            2: "#e6adc0",
            3: "#93c4d5",
            4: "#bad0b9",
            5: "#cbb2ff"
        }

        self.correct_colors = {
            "correct": "#f78c1b", 
            "incorrect": "#1b7ef7"
        }

        self.exp_data_hf[setup].map(lambda x, idx: self.create_sankey_link(x, idx, setup, sent_color), with_indices=True)

        self.sources = [src for src, _, _ in self.links.keys()]
        self.targets = [tgt for _, tgt, _ in self.links.keys()]
        self.values = list(self.links.values())

        acc_per_turn = []
        for i in range(0, len(self.lbls)-2, 2):
            sum_a = sum([v for k, v in self.links.items() if k[0] == i])
            sum_b = sum([v for k, v in self.links.items() if k[0] == i+1])
            total_beliefs = sum_a + sum_b
            
            if total_beliefs != 0:
                perc_correct = round((sum_a/total_beliefs)*100, 2)

                acc_per_turn.append(perc_correct)       

        avg_acc = round(float(self.mean_preds_df.loc[setup]["accuracy"])*100, 2)
        acc_per_turn.append(avg_acc)

        while len(acc_per_turn) < 11:
            acc_per_turn.append(None)
        
        return acc_per_turn

    def plot_dist_split(self):
        """
        This function plots Fig. 2, i.e., distributed knowledge results for llama and qwen in different subplots.
        """
        #run_files = [["eval/dist_llama_roc.csv", "eval/dist_llama_cmu.csv"], ["eval/dist_qwen8_roc.csv", "eval/dist_qwen8_cmu.csv"], ["eval/dist_qwen_roc.csv", "eval/dist_qwen_cmu.csv"]]
        run_files = [["eval/dist_llama_cmu.csv", "eval/dist_llama_roc.csv"], ["eval/dist_qwen8_cmu.csv", "eval/dist_qwen8_roc.csv"], ["eval/dist_qwen32_cmu.csv", "eval/dist_qwen32_roc.csv"]]

        nrow = 3
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Improved': '#cbb2ff',
            'Single': "#17b2e1",  
            'Base': '#ffc7a2',
            'Random': 'lightgray'                    
        }

        for r in range(nrow):
            for c in range(ncol):
                df_experiment_files = pd.read_csv(run_files[r][c], delimiter = ";")

                # get avg of three runs
                output_0_acc = df_experiment_files["output_0"].map(lambda x: self.get_avg_metric(x, round_acc = True))
                output_1_acc = df_experiment_files["output_1"].map(lambda x: self.get_avg_metric(x, round_acc = True))
                output_2_acc = df_experiment_files["output_2"].map(lambda x: self.get_avg_metric(x, round_acc = True))

                s_acc = (output_0_acc + output_1_acc + output_2_acc)/3
                s_std = pd.DataFrame([output_0_acc, output_1_acc, output_2_acc]).std()

                # if "booksum" in self.task:
                #     s_acc[0] = 34.5 # keep same avg random boundary across 

                df_acc = pd.concat([df_experiment_files, s_acc.rename("accuracy"), s_std.rename("std")], axis=1)
                plot_df = df_acc.pivot(index='task', columns='framework', values='accuracy') 
                plot_df = plot_df.dropna(axis=1, how='all')

                err_df = df_acc.pivot(index="task", columns="framework", values="std")
                err_df = err_df.reindex(index=plot_df.index, columns=plot_df.columns)

                original_cols = plot_df.columns
                colors = [color_map[framework] for framework in original_cols]

                ax = axes[r][c]
                if c == 0:
                    ylabel = df_acc["model"][1]
                    yticks = np.arange(0, 101, 20)
                else:
                    ylabel = ""

                if r == 1:
                    title = None

                # df_acc["accuracy"].plot(
                #     ax=ax, 
                #     ylabel=ylabel, 
                #     yticks=yticks,
                #     color=colors, 
                #     legend = (c==1 and r==0),
                #     #figsize=(4.25,5.5), # w x h
                #     figsize=(4, 5.25), # w x h
                #     kind="bar",
                #     rot=0,
                #     xlabel = "",
                #     edgecolor='black',
                #     width = 0.85,
                #     yerr = df_acc["std"]
                # )
                plot_df.plot(
                    ax=ax, 
                    ylabel=ylabel, 
                    yticks=yticks,
                    color=colors, 
                    legend = (c==1 and r==0),
                    #figsize=(4.25,5.5), # w x h
                    figsize=(4,5.25), # w x h
                    kind="bar",
                    rot=0,
                    xlabel = "",
                    edgecolor='black',
                    width = 0.68,
                    yerr = err_df,
                    capsize = 2,
                    error_kw={"elinewidth": 0.8}
                )

                if r == 0 and c == 1:
                    ax.legend(title=None, loc="upper left", fontsize=8)

                if c == 1:
                    ax.tick_params(labelleft=False)    

                for i, patch in enumerate(ax.patches):

                    #original_width = patch.get_width()
                    #new_width = original_width * 0.8 # Shrink bar to 80% to create a gap
                    #patch.set_width(new_width)

                    # --- ANNOTATION (VALUE) LOGIC ---
                    height = patch.get_height()
                    if not np.isnan(height) and height > 0:
                        ax.annotate(
                            f"{height:.1f}", # Format the accuracy to 2 decimal places
                            (patch.get_x() + patch.get_width() / 2, height), 
                            ha='center',                                     
                            va='bottom',                                     
                            xytext=(0, 3),                                   
                            textcoords='offset points',
                            rotation=0,                                     
                            fontsize=6
                        )

        plt.tight_layout() 
        #plt.subplots_adjust(wspace=0., hspace=0.)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.savefig(f"visualizations/dist_split.pdf")

    def compute_std(self):
        """
        This method computes the standard deviation between runs.
        """
        run_file = "eval/final.csv"

        df_experiment_files = pd.read_csv(run_file, delimiter = ";")

        # get avg of three runs
        output_0_acc = df_experiment_files["output_0"].map(lambda x: self.get_avg_metric(x))
        output_1_acc = df_experiment_files["output_1"].map(lambda x: self.get_avg_metric(x))
        output_2_acc = df_experiment_files["output_2"].map(lambda x: self.get_avg_metric(x))

        s_acc = round(((output_0_acc + output_1_acc + output_2_acc)/3), 2)

        df_std = round(pd.concat([output_0_acc, output_1_acc, output_2_acc], axis = 1).std(axis=1), 2)

        df_acc = pd.concat([
            df_experiment_files,
            output_0_acc.rename("accuracy_0"),
            output_1_acc.rename("accuracy_1"),
            output_2_acc.rename("accuracy_2"),
            s_acc.rename("accuracy"), 
            df_std.rename("std")],
            axis=1)

        df_acc.to_csv("visualizations/final.csv")

    def compute_all_metrics(self, exp_file: str = "./eval/final.csv", instances: int = 1000):
        """
        This method computes all metrics for all main experiments.
        """
        df_experiment_files = pd.read_csv(exp_file, delimiter = ";")    

        self.last = False

        # get result dicts for all three runs
        res_0 = df_experiment_files["output_0"].map(lambda x: self.get_avg_metrics(x, instances=instances))
        res_1 = df_experiment_files["output_1"].map(lambda x: self.get_avg_metrics(x, instances=instances))
        res_2 = df_experiment_files["output_2"].map(lambda x: self.get_avg_metrics(x, instances=instances))

        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau", "conv_length_faa", "stop", "stop_percentage", "instances", "wrong_format", "zero_counter", "zero_percentage", "double_counter", "double_percentage"]
        dict_output_0 = defaultdict(list)
        dict_output_1 = defaultdict(list)
        dict_output_2 = defaultdict(list)
        for metric in self.metrics:
            for row in res_0.items():
                if metric not in ["conv_length_faa", "stop", "instances", "wrong_format", "zero_counter", "double_counter"]:
                    dict_output_0[metric].append(round(row[1][metric]*100, 2))
                    dict_output_1[metric].append(round(res_1[row[0]][metric]*100, 2))
                    dict_output_2[metric].append(round(res_2[row[0]][metric]*100, 2))
                else:
                    dict_output_0[metric].append(row[1][metric])
                    dict_output_1[metric].append(res_1[row[0]][metric])
                    dict_output_2[metric].append(res_2[row[0]][metric])

        output_0 = pd.DataFrame.from_dict(dict_output_0)
        output_1 = pd.DataFrame.from_dict(dict_output_1)
        output_2 = pd.DataFrame.from_dict(dict_output_2)
        # compute average
        s_metrics = round(((output_0 + output_1 + output_2)/3), 2)

        df_metrics = pd.concat([
            df_experiment_files,
            s_metrics
            ],
            axis=1)

        df_metrics.to_csv("visualizations/all_final.csv", sep = ";")

    def plot_last(self):
        ''' 
        This method prints the accuracy of the final answer agent and of the last "player".
        '''
        df_experiment_files = pd.read_csv("eval/args_hypo_runs_llama_cmu.csv", delimiter = ";")

        self.last = False
        df_experiment_files["accuracy"] = df_experiment_files["run"].map(self.get_avg_metric)

        self.last = True
        df_experiment_files["last_accuracy"] = df_experiment_files["run"].map(self.get_avg_metric)

        print(df_experiment_files)

    def plot_roc_metrics_single_run(self, exp_file: str = "", instances: int = 1000):
        '''
        This method prints all secondary metrics for ROCStories on distributed base results (single vs. base vs. improved).
        '''
        df_experiment_files = pd.read_csv(exp_file, delimiter = ";")

        self.task = "rocstories"
        self.last = False
        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]
        for metric in self.metrics:
            df_experiment_files[metric] = df_experiment_files["output"].map(lambda x: vis.get_avg_metric(x, metric=metric, round_acc=True, instances=instances))

        print(df_experiment_files)
        df_experiment_files.to_csv(exp_file.replace("eval", "visualizations"), sep = ";")

    def plot_roc_metrics_three_runs(self, exp_file: str = "eval/final.csv", instances: int = 1000):
        '''
        This method prints all secondary metrics for ROCStories on distributed base results (single vs. base vs. improved).
        '''
        df_experiment_files = pd.read_csv(exp_file, delimiter = ";") 

        df_roc_files = df_experiment_files.where(df_experiment_files["task"]=="StoryCon").dropna()   

        self.task = "rocstories"
        self.last = False
        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]

        output_0 = pd.DataFrame()
        output_1 = pd.DataFrame()
        output_2 = pd.DataFrame()
        for metric in self.metrics:
            # get avg of three runs
            output_0[metric] = df_roc_files["output_0"].map(lambda x: self.get_avg_metric(x, metric=metric, round_acc=True, instances=instances))
            output_1[metric] = df_roc_files["output_1"].map(lambda x: self.get_avg_metric(x, metric=metric, round_acc=True, instances=instances))
            output_2[metric] = df_roc_files["output_2"].map(lambda x: self.get_avg_metric(x, metric=metric, round_acc=True, instances=instances))

        s_metrics = round(((output_0 + output_1 + output_2)/3), 2)

        df_metrics = pd.concat([
            df_roc_files,
            s_metrics
            ],
            axis=1)

        df_metrics.to_csv("visualizations/secondary_roc.csv", sep = ";")

    def plot_dist_sanity_checks(self):
        '''
        This method prints sanity checks for the distributed base experiments (single vs. base vs. improved).
        '''
        df_experiment_files = pd.read_csv("eval/distributed_base_runs.csv", delimiter = ";")

        self.last = False
        df_experiment_files["wrong_format"] = df_experiment_files["output"].map(lambda x: vis.get_sanity_checks(x))

        print(df_experiment_files)
        df_experiment_files.to_csv("./visualizations/distributed_base_sanity.csv", sep = ";") # multi:stimmt nicht

    def plot_args_hypos_metrics(self):
        '''
        This method prints the final and last accuracy for the argumentative reasoning experiments (args vs. hypo vs. full).
        '''
        self.task = "rocstories"
        df_experiment_files = pd.read_csv("eval/args_hypo_runs_llama_roc.csv", delimiter = ";")

        self.last = False
        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]
        for metric in self.metrics:
            df_experiment_files[metric] = df_experiment_files["run"].map(lambda x: vis.get_avg_metric(x, metric=metric))

        self.last = True
        df_experiment_files["last_accuracy"] = df_experiment_files["run"].map(self.get_avg_metric)
        
        print(df_experiment_files)
        df_experiment_files.to_csv("./visualizations/rocstories/args_hypos_llama.csv", sep = ";")

    def get_exp_groups(self):
        '''
        This method groups all runs that differ only by one configuration (e.g., final answer agent vs. no final answer agent) 
        '''
        df_runs = pd.read_csv("logs/runs.csv", delimiter = ";")

        # important keys for comparison
        keys = ["topology", "do_sample", "version_idx", "top_p", "num_beams", "length_penalty", "final_reorder", "candidate", "names", "temperature", "top_k", "framework", "output_format", "conv_format", "thinking_budget", "max_new_tokens", "variation", "min_p", "quantized", "final", "torch_dtype"]
        for row_idx, row in df_runs.iterrows():
            if row["split"] != "mini":
                row_data = dict(row)

                run_groups = defaultdict(list)

                for next_idx, next_row in df_runs.iterrows():
                    if next_row["split"] != "mini":
                        if next_idx != row_idx:
                            distinct_vals = 0
                            next_row_data = dict(next_row)

                            found_key = ""
                            for key, row_val in row_data.items():
                                if key in keys:
                                    #check = False
                                    check_row = self.check_val(row_val)
                                    check_next = self.check_val(next_row_data[key])

                                    if (check_row or check_next) and row_val != next_row_data[key]:
                                        distinct_vals += 1
                                        found_key = key

                            if distinct_vals == 1:
                                if row_idx not in run_groups[found_key]:
                                    run_groups[found_key].append(row_idx)
                                
                                run_groups[found_key].append(next_idx)

                if run_groups != {}:
                    for k, ids in run_groups.items():
                        df_objects = []
                        for idx in ids:
                            df_obj = df_runs.iloc[idx]

                            df_objects.append(df_obj)

                        if df_objects != []:
                            df_compare = pd.concat(df_objects, axis = 1)

                            path = f"./visualizations/comp_runs/{k}/"
                            os.makedirs(path, exist_ok=True)
                            file = f"{path}{"_".join([str(df_obj["run_idx"]) for df_obj in df_objects])}.csv"

                            df_compare.to_csv(file, sep=";")

    def check_val(self, val):
        '''
        This method checks whether a value is a float value.

        @param
            val: checked value
        @returns: True if value is not float
        '''
        if type(val) != float:
            return True
        elif type(val) == float and not math.isnan(val):
            return True
        else: 
            return False

    def plot_topology_changes(self, topology: str = "Chain"):
        """
        This method creates the plots for Fig. 7, i.e., the intermediate micro precision per topology.

        @params
            topology: Chain/Layered/Star
        """
        df_experiment_files = pd.read_csv("eval/topologies_dev.csv", delimiter = ";")

        df_top_files = df_experiment_files.where(df_experiment_files["framework"]==topology).dropna()

        acc_per_turn = self.plot_acc_changes(df_experiment_files=df_top_files, instances=250, one_model = True)
        df_acc_changes = pd.DataFrame()
        if topology == "Star":
            self.num_turns = 2
            df_acc_changes[0] = acc_per_turn.loc[0:4].mean()
            df_acc_changes[1] = acc_per_turn.loc[5]
        
        df_acc_changes.T.plot(xticks=np.arange(self.num_turns), yticks = np.arange(0, 101, 10), ylabel = "Micro precision", xlabel = "Turn")

        plt.savefig(f"./visualizations/{topology}_changes.pdf")
        
if __name__ == "__main__":
    # experiment_files = {
    #     "single": "outputs/single/Qwen3-32B/rocstories/test/outputs_360",
    #     "correct": "outputs/multi/Qwen3-32B/rocstories/test/outputs_486_487",
    # }
    experiment_files = {}
    vis = Vis(experiment_files=experiment_files, task = "rocstories", variation = None, last = False)
    #vis.perform_ttest(comparison="single", exp_name="correct_qwen32_2")
    #vis.perform_paired_bootstrap()

    #vis.compute_all_metrics("./eval/final.csv")

    # Create Table 13 -> ./visualizations/rocstories/secondary_roc.csv
    #vis.plot_roc_metrics_three_runs()

    # Create Figure 2 -> ./visualizations/dist_split.pdf
    vis.plot_dist_split() 

    # Create Figure 4a -> ./visualizations/args_hypos.pdf
    #vis.plot_arg_hypo_changes()

    # Create Figure 4b -> ./visualizations/exploration.pdf
    #vis.plot_exploration_changes()

    # Create Figure 4c -> ./visualizations/correctness.pdf
    #vis.plot_correctness_changes()
