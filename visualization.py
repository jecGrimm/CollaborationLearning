from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import re
import ast
from pySankey.sankey import sankey
import numpy as np
from evaluate import Evaluator, evaluate
import plotly.graph_objects as go # TODO: Full sankey with go https://python-graph-gallery.com/sankey-diagram-with-python-and-plotly/
import statistics
from scipy.stats.mstats import ttest_ind
from matplotlib import colors
from matplotlib.patches import Patch

class Vis():
    def __init__(self, experiment_files: dict = {}, variation: bool = True, conv_format: str = "", last: bool = False, task: bool = "rocstories"):
        self.experiment_files = experiment_files

        self.last = last

        self.conv_format = conv_format

        self.task = task
        self.variation = variation
        self.evaluator = None

        if task == "rocstories":
            if self.variation:
                self.vispath = f"visualizations/{self.task}/variation/{conv_format}"
            else:
                self.vispath = f"visualizations/{self.task}/no_variation/{conv_format}"
        else:
            self.vispath = f"visualizations/{self.task}/{conv_format}"
        os.makedirs(self.vispath, exist_ok = True)

        if self.last:
            self.last_app = "_last"
        else:
            self.last_app = ""

        if experiment_files != {}:
            self.initialize_experiment_files()

    def initialize_experiment_files(self):
        self.exp_data_hf = {k: load_from_disk(v) for k, v in self.experiment_files.items()}

        self.create_eval_dfs()


    def create_eval_dfs(self):
        if self.last:
            eval_files = defaultdict(str)
            for k, v in self.experiment_files.items():
                if "single" in k:
                    print("Setup is single, not performing last evaluation!")
                    eval_files[k] = f"{v.replace('outputs', 'eval')}.jsonl"
                else:
                    eval_files[k] = f"{v.replace('outputs', 'eval')}_last.jsonl"
            #eval_files = {k: f"{v.replace("outputs", "eval")}_last.jsonl" for k, v in experiment_files.items()}
        else:
            eval_files = {k: f"{v.replace('outputs', 'eval')}.jsonl" for k, v in self.experiment_files.items()}

        self.eval_data_hf = {k: self.load_eval_data(v, k) for k, v in eval_files.items()}

        mean_preds = defaultdict(list)

        if self.task == "rocstories":
            self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]
        else:
            self.metrics = ["accuracy"]

        for setup, eval_data in self.eval_data_hf.items():
            for metric in self.metrics:
                mean_preds[setup].append(float(eval_data.with_format("pandas")[metric].mean()))

        self.mean_preds_df = pd.DataFrame(mean_preds.values(), index=mean_preds.keys(), columns=self.metrics) 


    def load_eval_data(self, eval_file, exp_name):
        try:
            return load_dataset("json", data_files=eval_file, split="train")
        except:
            fw = ""
            if "single" in eval_file:
                fw = "single"
            else:
                fw = "multi"

            evaluator = Evaluator(self.experiment_files[exp_name], fw, last = self.last)
            
            if self.last and fw == "multi":
                evaluator.evaluate_last_beliefs()
            else:
                evaluator.evaluate_sent_order()
            return load_dataset("json", data_files=eval_file, split="train")

    def plot_performance(self):
        for metric in self.metrics:
            if self.variation:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: with variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 

                ax.bar_label(ax.containers[0])

                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}.png")
            else:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: without variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 
                ax.bar_label(ax.containers[0])
                
                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}.png")
            
            plt.close()

    def create_belief_instance_roc(self, index: int, setup: str):
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
        out = self.exp_data_hf[setup].select([index])[0]
        eval = self.eval_data_hf[setup].select([index])[0]

        beliefs = []
        if "turns" in out.keys():
            for i, turn in enumerate(out["turns"]):
                conv_list = self.extract_structured_answer(turn)

                player = out["player"][i]

                if type(conv_list) == list:
                    for conv_dir in conv_list:
                        if type(conv_dir) == dict and conv_dir["title"] in out["titles"] and "label" in conv_dir.keys():                        
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
        

    def plot_instance(self, index: int, setup: str):

        #vis_beliefs_df = pd.DataFrame(vis_beliefs)
        if self.task == "rocstories":
            beliefs = self.create_belief_instance_roc(index, setup)
        else:
            beliefs = self.create_belief_instance_cmu(index, setup)
        belief_df = pd.DataFrame(beliefs)
        belief_pivot = belief_df.pivot(columns = ["correct"], values=["turn", "label"])

        styler = belief_pivot.style
        styler.hide(axis = 0)\
            .highlight_null(props="color: transparent;")\
            .set_properties(props="color: transparent;")

        # styled_beliefs = styled_beliefs.style.highlight_null(props="color: transparent;")
        # styled_beliefs = styled_beliefs.hide_index()
        #styled_beliefs = styled_beliefs.set_index("turn")

        html = styler.to_html()
        with open(f"{self.vispath}/instance.html", "w", encoding = "utf-8") as file:
            file.write(html)
        #dfi.export(styled_beliefs, "test.png")

    def create_conv_list_roc(self, conv_str: str):
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
        if "argument" in item:
            item = item.strip("argument")
        elif "position" in item:
            item = item.strip("position")
        elif "label" in item:
            item = item.strip("label")
        return item
        
    def plot_sankey(self, setup: str):
        # last hypo == correct/incorrect -> final hypo == correct/incorrect 
        self.sankey_inst = []
        self.exp_data_hf[setup].map(lambda x, idx: self.create_sankey_hf(x, idx, setup), with_indices=True)

        #df = pd.read_csv(url, sep=" ", names=["true", "predicted"])
        df = pd.DataFrame(self.sankey_inst, columns = ["last_hypo", "final_hypo"])

        colors = {
            "correct": "#f78c1b",
            "incorrect": "#1b7ef7",
        }

        sankey(df["last_hypo"], df["final_hypo"], aspect=20, colorDict=colors, fontsize=12)

        fig = plt.gcf()

        if self.variation:
            fig.savefig(f"{self.vispath}/sankey_{setup}_var.png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(f"{self.vispath}/sankey_{setup}_no_var.png", bbox_inches="tight", dpi=150)

    def create_sankey_hf(self, x, idx, setup):
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

    def create_full_sankey_hf(self, x, idx, setup):
        """
        @deprecated
        """
        beliefs = self.create_belief_instance(idx, setup)

        sent_beliefs = [["not mentioned" for _ in range(self.num_turns)] for _ in range(5)]

        turn_idx = 0
        for belief in beliefs:
            if setup == "chain":
                turn_idx = belief["turn"]-1
            elif setup == "star":
                if belief["agent"] != "Player 6":
                    turn_idx = 0
                else:
                    turn_idx = 1
            elif setup == "graph":
                if belief["turn"] < 6:
                    turn_idx = 0
                elif belief["turn"] > 6 and belief["agent"] != "Player 6":
                    turn_idx = 1
                else:
                    turn_idx = 2

            sent_beliefs[belief["gold_sent"]-1][turn_idx] = belief["correct"]

            # TODO:graph
        self.sankey_inst += sent_beliefs

    def plot_sankey_full(self, setup: str):
        """
        @deprecated
        """
        # last hypo == correct/incorrect -> final hypo == correct/incorrect 

        #df = pd.read_csv(url, sep=" ", names=["true", "predicted"])
        self.num_turns = 0
        if setup == "chain":
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif setup == "star":
            self.num_turns = 2
        elif setup == "graph":
            self.num_turns = 3

        columns = [f"Turn {i}" for i in range(1, self.num_turns+1)]
        self.sankey_inst = []

        self.exp_data_hf[setup].map(lambda x, idx: self.create_full_sankey_hf(x, idx, setup), with_indices=True)
        
        df = pd.DataFrame(self.sankey_inst, columns = columns) # TODO: define column names

        colors = {
            "correct": "#f78c1b",
            "incorrect": "#1b7ef7",
            "not mentioned": "#f5ee27"
        }

        sankey(df["last_hypo"], df["final_hypo"], aspect=20, colorDict=colors, fontsize=12)

        fig = plt.gcf()

        if self.variation:
            fig.savefig(f"{self.vispath}/sankey_{setup}_var__{self.conv_format}.png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(f"{self.vispath}/sankey_{setup}_no_var__{self.conv_format}.png", bbox_inches="tight", dpi=150)

    def plot_sankey_full_plotly(self, setup, sent_color):        
        self.num_turns = 0
        if "chain" in setup:
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif "star" in setup:
            self.num_turns = 2
        elif "graph" in setup:
            self.num_turns = 3

        # self.sources = []
        # self.targets = []
        # self.values = [] # sentence
        self.lbls = [f"{lbl}_{i}" for i in range(self.num_turns) for lbl in ["correct", "incorrect"]]
        colors = [color for i in range(self.num_turns) for color in ["#f78c1b", "#1b7ef7"]]
        #positions = []

        #self.lbls = [str(i) for i in range(self.num_turns)]
        #colors = [color for i in range(self.num_turns) for color in ["orange", "blue"]]
        
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
        last_perc_a = 0.0
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
                # if perc_a > last_perc_a:
                #     percentage_lbls.append(f"{self.lbls[i]}: +{round(perc_a - last_perc_a, 2)}")
                # else:
                #     percentage_lbls.append(f"{self.lbls[i]}: -{round(last_perc_a - perc_a, 2)}%")

                if perc_b > last_perc_b:
                    percentage_lbls.append(f"{self.lbls[i+1]}: +{round(perc_b - last_perc_b, 2)}")
                else:
                    percentage_lbls.append(f"{self.lbls[i+1]}: -{round(last_perc_b - perc_b, 2)}%")

            last_perc_a = perc_a
            last_perc_b = perc_b

        perc_a = round(float(self.mean_preds_df.loc[setup]["accuracy"])*100, 2)
        perc_b = 100-perc_a
        percentage_lbls.append(f"{self.lbls[i]}: {perc_a}%")
        # if perc_a > last_perc_a:
        #     percentage_lbls.append(f"{self.lbls[i].split('_')[0]}: +{round(perc_a - last_perc_a, 2)}")
        # else:
        #     percentage_lbls.append(f"{self.lbls[i].split('_')[0]}: -{round(last_perc_a - perc_a, 2)}%")

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

            if sent_color:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_var_sent.html")
            else:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_var.html")
        else:
            fig.update_layout(title_text=f"{setup} without variation: {self.conv_format}", font_size=10)
            fig.show()
            if sent_color:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_no_var.html")
            else:
                fig.write_html(f"{self.vispath}/full_sankey_{setup}_no_var_sent.html")
        
    def create_sankey_link(self, inst, idx, setup, sent_color):
        beliefs = self.create_belief_instance_roc(idx, setup)

        #sent_beliefs = [["not mentioned" for _ in range(self.num_turns)] for _ in range(5)]
        for sent_idx in range(1, 6):
            sent_beliefs = [belief for belief in beliefs if belief["gold_sent"] == sent_idx]
            turn_idxs = [(belief_idx, self.get_turn_idx(setup, belief)) for belief_idx, belief in enumerate(sent_beliefs)]

            for belief_idx, turn_idx in turn_idxs:
                curr_belief = sent_beliefs[belief_idx]

                if turn_idx < turn_idxs[-1][1]:
                    next_idx = [next_item[1] for next_item in turn_idxs if next_item[1] > turn_idx][0]

                    next_targets = [item for item in turn_idxs if item[1] == next_idx]

                    # start sources for next beliefs
                    for target_idx, target_turn in next_targets:
                        source_lbl = f"{curr_belief['correct']}_{turn_idx}"
                        #self.sources.append(self.lbls.index(source_lbl))
                        #self.sources.append(turn_idx)

                        # if curr_belief["correct"] == "correct":
                        #     self.values.append("correct")
                        # else:
                        #     self.values.append("incorrect")

                        target_belief = sent_beliefs[target_idx]
                        target_lbl = f"{target_belief['correct']}_{target_turn}"
                        #self.targets.append(self.lbls.index(target_lbl))
                        #self.targets.append(target_turn)

                        if sent_color:
                            self.links[(self.lbls.index(source_lbl), self.lbls.index(target_lbl), self.sent_colors[sent_idx])] += 1
                        else: 
                            self.links[(self.lbls.index(source_lbl), self.lbls.index(target_lbl), self.correct_colors[curr_belief["correct"]])] += 1

    def get_target(self, belief_idx, curr_turn_idx, beliefs, setup):
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
        turn_idx = 0
        if "chain" in setup:
            turn_idx = belief["turn"]-1
        elif "star" in setup:
            if belief["agent"] != "Player 6":
                turn_idx = 0
            else:
                turn_idx = 1
        elif "graph" in setup:
            if belief["turn"] < 6:
                turn_idx = 0
            elif belief["turn"] > 6 and belief["agent"] != "Player 6":
                turn_idx = 1
            else:
                turn_idx = 2
        
        return turn_idx
    
    def perform_ttest(self, comparison = "single", exp_name = ""):
        first_accs = [list(eval_data['accuracy']) for setup, eval_data in self.eval_data_hf.items() if comparison in setup][0]

        stats_dict = defaultdict(list)
        for setup, eval_data in self.eval_data_hf.items():
            if comparison not in setup:
                second_accs = list(eval_data['accuracy'])
                
                stats = ttest_ind(first_accs, second_accs, equal_var=False, alternative="less")

                stats_dict["first_setup"].append(comparison)
                stats_dict["first_mean"].append(np.mean(first_accs))
                stats_dict["second_setup"].append(setup)
                stats_dict["second_mean"].append(np.mean(second_accs))
                stats_dict["Welch's t-test"].append(stats.statistic)
                stats_dict["p-value"].append(stats.pvalue)

                if stats.pvalue < 0.05:
                    stats_dict["significant"] = True
                else:
                    stats_dict["significant"] = False

        stats_df = pd.DataFrame(stats_dict)

        outfile = f"{self.vispath}/significance_{comparison}"
        if exp_name != "":
            outfile += f"_{exp_name}"

        if self.last:
            outfile += f"_last"

        outfile += ".csv"
        stats_df.to_csv(outfile, sep = ";")

    def correlate_last_final(self, setup):
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

        # fig, ax = plt.subplots()
        # #colors = ["#ffc7a2", "#e6adc0", "#93c4d5", "#bad0b9"]
        label = [mean_final.index[i].split("_")[1] for i in range(len(mean_final.index))]
        #label = ["hypo", "full", "reasons", "support"]*2
        df = pd.concat([mean_final["accuracy"].rename("final"), mean_last["accuracy"].rename("last")], axis = 1)

        print(df.corr())
        # nrof_labels = len(label)  
        # colors = colors = cm.rainbow(np.linspace(0, 1, len(d['f']))) # create a bunch of colors

        # for i, r in df.iterrows():
        #     ax.plot(r['last'], r['final'], 'o', markersize=10, color=colors[i], linewidth=0.1, label=label[i])

        # #ax = df.plot.scatter(x="last", y = "final", c = ["#ffc7a2", "#e6adc0", "#93c4d5", "#bad0b9"], label = ["hypo", "full", "reasons", "support"])
        # plt.legend(loc = "best")

        colorlist = list(colors.ColorConverter.colors.keys())
        fig, ax = plt.subplots()
        [df.iloc[[i]].plot.scatter('last', 'final', ax=ax, s=50, label=l,
                                color=colorlist[i % len(colorlist)])
        for i,l in enumerate(label)]
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        #setup = mean_final.index[0].split("_")[1]
        #setup = "all"
        ax.set_title(f"Correlation between MAS and final performance: {setup}")

        plt.savefig(f"{self.vispath}/last_final_correlation_{setup}.png")
        plt.close()

    def plot_all_table(self):
        df_experiment_files = pd.read_csv(f"eval/args_hypos_runs_llama_roc.csv", delimiter = ";").set_index("format")

        df_acc = df_experiment_files.map(self.get_avg_acc)

        # plot sankeys
        if self.last == False:
            df_dict = df_experiment_files.T.to_dict()
            acc_changes = defaultdict(list)
            for conv_format, exp_files in df_dict.items():
                self.__init__(experiment_files = exp_files, variation = self.variation, conv_format = conv_format, last = self.last)
                
                for setup in exp_files.keys():
                    if "single" not in setup and "random" not in setup and "du" not in setup:
                        #self.plot_sankey_full_plotly(setup, sent_color=False)

                        acc_changes[f"{setup}_{conv_format}"] = self.get_acc_changes(setup) # TODO: Test
                    else:
                        acc_changes[f"{setup}"] = [float(df_acc[setup][conv_format])]*11

            df_acc_changes = pd.DataFrame(acc_changes)
            df_acc_changes.plot()
        
        df_acc.to_csv(f"./visualizations/{self.task}/all{self.last_app}.csv", sep = ";")


    def plot_acc_changes(self, run_file):
        df_experiment_files = pd.read_csv(run_file, delimiter = ";")

        df_experiment_files["accuracy"] = df_experiment_files["run"].map(self.get_avg_acc)

        # plot sankeys
        df_dict = df_experiment_files.T.to_dict()
        acc_changes = defaultdict(list)
        for row in df_dict.values():
            conv_format = row["format"]
            task = row["task"]
            exp_file = row["run"]
            setup = conv_format
            self.__init__(experiment_files = {setup: exp_file}, variation = self.variation, conv_format = conv_format, last = self.last, task = task.lower())
            
            self.evaluator = Evaluator(exp_file,"multi", task.lower())

            self.num_turns = max([len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))])-1
            self.corr_per_turn = [0]*self.num_turns
            self.beliefs_per_turn = [0]*self.num_turns
            self.exp_data_hf[setup] = self.exp_data_hf[setup].map(lambda x, idx: self.get_acc_per_turn(x, idx, setup), with_indices=True)

            acc_per_turn = [0]*self.num_turns
            for i in range(self.num_turns):
                if self.beliefs_per_turn[i] != 0:
                    acc_per_turn[i] = round((self.corr_per_turn[i]/self.beliefs_per_turn[i])*100, 2)
            
            #acc_per_turn[-1] = row["accuracy"]
            acc_changes[setup] = acc_per_turn

        df_acc_changes = pd.DataFrame(acc_changes)

        color_map = {
            'Hypo': '#cbb2ff',
            'Full': '#93c4d5',  
            'Args': '#ffc7a2'               
        }
        #df_acc_changes.plot(ylabel="Accuracy", xlabel="Turn", xticks = np.arange(self.num_turns), yticks=np.arange(0, 101, 10), color=list(color_map.values()))
        
        #plt.savefig(f"visualizations/{self.task}/args_hypos.png")
        #df_acc.to_csv(f"./visualizations/{self.task}/acc_changes_llama_roc.csv", sep = ";")

        return df_acc_changes

    def plot_arg_hypo_changes(self):
        llama_roc = self.plot_acc_changes("eval/args_hypo_runs_llama_roc.csv")
        llama_cmu = self.plot_acc_changes("eval/args_hypo_runs_llama_cmu.csv")

        df_list = [llama_roc, llama_cmu]
        
        nrow = 1
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Hypo': '#cbb2ff',
            'Full': '#93c4d5',  
            'Args': '#ffc7a2'               
        }
        subtitle = ["ROCStories", "CMU_Booksum"]
        for c in range(ncol):
            if c == 0:
                ylabel = "Micro Accuracy"
                xticks = np.arange(10)
            else:
                ylabel = ""
                xticks = np.arange(6)

            df_list[c].plot(
                ax=axes[c], 
                ylabel=ylabel, 
                xlabel="Turn", 
                xticks = xticks, 
                yticks=np.arange(0, 61, 10), 
                color=list(color_map.values()), 
                title = subtitle[c], 
                legend = (c==0),
                figsize=(4.5,2.5)
            )

        plt.savefig(f"visualizations/args_hypos.png")

    def get_acc_per_turn(self, inst, idx, setup):
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
        
    def get_avg_acc(self, exp_file):
        if self.last and "single" not in exp_file:
            eval_file = f"{exp_file.replace('outputs', 'eval')}_last.jsonl"
        else:
            eval_file = f"{exp_file.replace('outputs', 'eval')}.jsonl"
        
        try:
            eval_data = load_dataset("json", data_files=eval_file, split="train")
            acc = float(eval_data.with_format("pandas")["accuracy"].mean())
        except:
            if self.last and "single" not in exp_file:
                acc_mean, acc_last = evaluate(exp_file, framework = eval_file.split("/")[1], task = self.task, structured = True)
                acc = acc_last
            else:
                acc_mean, acc_last = evaluate(exp_file, framework = eval_file.split("/")[1], task = self.task, structured = False)
                acc = acc_mean

        return round(acc*100, 2)
    
    def get_acc_changes(self, setup, sent_color = False):  
        self.num_turns = 0
        if "chain" in setup:
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif "star" in setup:
            self.num_turns = 2
        elif "graph" in setup:
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
            
            perc_correct = round((sum_a/total_beliefs)*100, 2)

            acc_per_turn.append(perc_correct)       

        avg_acc = round(float(self.mean_preds_df.loc[setup]["accuracy"])*100, 2)
        acc_per_turn.append(avg_acc)

        while len(acc_per_turn) < 11:
            acc_per_turn.append(None)
        
        return acc_per_turn

    def plot_distributed_base(self):
        df_experiment_files = pd.read_csv(f"eval/distributed_base_runs.csv", delimiter = ";")

        s_acc = df_experiment_files["output"].map(self.get_avg_acc)
        df_acc = pd.concat([df_experiment_files, s_acc.rename("accuracy")], axis=1)

        # 1. Pivot the data to group by Task on X-axis
        plot_df = df_acc.pivot(index='task', columns=['model', 'framework'], values='accuracy')
        plot_df = plot_df.dropna(axis=1, how='all')

        # 2. Assign styles (Colors for Models, Hatches for Frameworks)
        color_map = {
            'Random': 'lightgray', 
            'Llama-3.1-8B-Instruct': '#93c4d5',  
            'Qwen3-32B': '#ffc7a2'               
        }

        hatch_map = {
            'random': '',      # Solid
            'Single': '',      # Solid
            'MAS': '//'        # Diagonal lines
        }

        original_cols = plot_df.columns
        colors = [color_map[model] for model, framework in original_cols]

        # 3. Plot the base colored bars (edgecolor is required for hatches to show)
        ax = plot_df.plot(
            kind="bar", 
            figsize=(6, 4.5),
            color=colors,
            width=0.85, 
            yticks=np.arange(0, 101, 10), 
            ylabel="Accuracy", 
            rot=0,
            edgecolor='grey',
            xlabel=""
        )

        # 4. Apply spacing, hatches, and accuracy value labels
        num_tasks = len(plot_df.index)
        for i, patch in enumerate(ax.patches):
            
            # --- SPACING LOGIC ---
            original_width = patch.get_width()
            new_width = original_width * 0.8 # Shrink bar to 80% to create a gap
            patch.set_width(new_width)
            
            # Re-center the bar
            new_x = patch.get_x() + (original_width - new_width) / 2
            patch.set_x(new_x)

            # --- HATCH LOGIC ---
            col_idx = i // num_tasks
            framework = original_cols[col_idx][1]
            
            hatch_pattern = hatch_map.get(framework, '')
            if hatch_pattern:
                patch.set_hatch(hatch_pattern)

            # --- ANNOTATION (VALUE) LOGIC ---
            height = patch.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}",                                 # Format the accuracy to 2 decimal places
                    (patch.get_x() + patch.get_width() / 2, height), 
                    ha='center',                                     
                    va='bottom',                                     
                    xytext=(0, 3),                                   
                    textcoords='offset points',
                    rotation=0,                                     
                    fontsize=10
                )

        # Extend the Y-axis slightly so the highest text labels don't get cut off
        #ax.set_ylim(0, 120)

        # 5. Create a combined legend for Models and Frameworks
        legend_elements = [
            # Model colors
            Patch(facecolor=color_map['Random'], edgecolor='black', label='Model: Random'),
            Patch(facecolor=color_map['Llama-3.1-8B-Instruct'], edgecolor='black', label='Model: Llama-3.1-8B-Instruct'),
            Patch(facecolor=color_map['Qwen3-32B'], edgecolor='black', label='Model: Qwen 3-32B'),
            
            # Empty patch for spacing in the legend
            Patch(facecolor='none', edgecolor='none', label=''), 
            
            # Framework patterns (represented with white boxes)
            Patch(facecolor='white', edgecolor='black', hatch='', label='Framework: Single'),
            Patch(facecolor='white', edgecolor='black', hatch='//', label='Framework: MAS')
        ]

        ax.legend(handles=legend_elements, title = "Model & Setup", loc="upper left")

        plt.tight_layout()
        plt.show()

        plt.savefig("./visualizations/distributed_base.png")
        #df_acc.plot(kind="bar", x = "task", y = "accuracy", yticks=np.arange(0, 110, 10), xlabel="Task", ylabel="Accuracy", rot = 0)
        df_acc.to_csv("./visualizations/distributed_base_results.csv", sep = ";")
        df_acc.to_latex()

if __name__ == "__main__":

    # for last in [False, True]:
    #     vis = Vis(task = "rocstories", variation = "True", conv_format = "", last = last)
    #     vis.plot_all_table()
    # experiment_files = {
    #     "single":"outputs/single/Llama-3.1-8B-Instruct/booksum/test/outputs_264",
    #     "MAS": "outputs/multi/Llama-3.1-8B-Instruct/cmu_booksum/test/outputs_265"
    # }
    vis = Vis()
    #vis.plot_distributed_base()
    vis.plot_arg_hypo_changes()
    #vis.plot_acc_changes("eval/args_hypo_runs_llama_roc.csv")

    #   vis.perform_ttest(comparison="single")

    print("Done")