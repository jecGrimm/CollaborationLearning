def play_games(self, batch, pipe):
    '''
    @deprecated
    This function processes conversations iteratively.

    @params
        batch: batch of instances
        pipe: pipeline object
    @returns
        dictionary with the conversational turns and player names
    '''
    batch_turns = []
    batch_players = []
    for i in range(len(batch["storyid"])):
        conv_round = 0
        turn = 0
        conversation = ""
        turns = []
        players = []

        roles = dict(zip(self.NAMES, batch["shuffled_sents"][i]))
        while conv_round < self.conversation_rounds:
            while turn < 5:
                player_name = self.NAMES[turn]

                model_out = pipe(
                        self.create_message(conv_round, turn, conversation, roles, player_name),
                        max_new_tokens=256
                )
                
                model_answer = model_out[0]["generated_text"][2]["content"]
            
                conversation += f"\n{player_name}: {model_answer}\n"
                #turns.append({"player": player_name, "answer": model_answer, "system_prompt": model_out[0]["generated_text"][0]["content"], "turn_prompt": model_out[0]["generated_text"][1]["content"]})
                turns.append(model_answer)
                players.append(player_name)
                if "STOP" in model_answer and conv_round > 0:
                    conv_round = self.conversation_rounds
                    turn = 5
                else: 
                    turn += 1
            conv_round += 1
        
        #batch_convs.append(turns)
        batch_turns.append(turns)
        batch_players.append(players)

    return {"turns": batch_turns, "players": batch_players}

def get_instances(outdir: str, rand_idx: int = None):
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
        rand_idx = random.randint(0, len(eval_data)-1)

    # select instances
    out = exp_data.select([rand_idx])[0]
    eval = eval_data.select([rand_idx])[0]

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

    def create_compare_dict(self, split):
        '''
        This method creates a dictionary that maps the run index to the corresponding evaluation file for a given split.
        '''
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

    roc_sents = set()
    with open("data/ROCStories_winter2017 - ROCStories_winter2017.csv", "r", encoding = "utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        for row in reader:
            roc_sents.add(row[1])
    print("intersect:", len(test_sents.intersection(roc_sents)))
    print("intersect:", val_sents.intersection(roc_sents))

def run_final_only(config: dict = None, logger: ExperimentLog = None):
    """
    @deprecated
    This function runs the experiment.

    @params
        config: configurations for the experiment
        logger: experiment logger
    """
    if logger == None:
        logger = ExperimentLog(config)

    outfile = config["experiment_file"]

    if not (os.path.isfile(outfile) or os.path.isdir(outfile)):
            multi = MultiAgent(model_id = config["model_id"], split = config["split"], conversation_rounds = config["conversation_rounds"], prompt_candidate = config["candidate"])

            logger.log_multi(multi)

            #multi.prompt_models(logger, device = config["device"], batch_size = config["batch_size"])
            multi.prompt_final_only(logger, device = config["device"], batch_size = config["batch_size"], conv_file = config["conv_file"], variation = config["variation"])

            outfile = multi.define_outfile(logger, hf = True)

    # evaluate
    evaluate_sent_order(outfile, config["framework"], logger)

    logger.store_run_data()

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

    def plot_topology_changes(self):
        "@deprecated, use plot_all_table instead"
        self.task = "rocstories"
        df_list = [self.plot_acc_changes("eval/multi/Qwen3-32B/rocstories/dev/topologies_250.csv")]
        
        nrow = 1
        ncol = 1
        fig, axes = plt.subplots(nrow, ncol)

        # plot counter
        color_map = {
            'Hypo': '#cbb2ff',
            'Full': "#17b2e1",  
            'Args': '#ffc7a2',
            'Reject': '#B2FFCB',
            'Support': '#FFB2E6',
            'Correct': '#C90D72',
            'Incorrect': '#0D72C9'                           
        }
        #subtitle = ["ROCStories", "CMU_Booksum"]
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
                #title = subtitle[c], 
                legend = (c==0),
                figsize=(4.5,2.5)
            )

        plt.savefig(f"visualizations/args_hypos/rocstories/variation/full_full/topologies.png")

    def plot_distributed_base(self):
        """
        This function creates a bar plot of the accuracy on both games for the baselines and unstructured MAS setups.
        """
        df_experiment_files = pd.read_csv(f"eval/distributed_base_runs.csv", delimiter = ";")

        output_0_acc = df_experiment_files["output_0"].map(self.get_avg_metric)
        output_1_acc = df_experiment_files["output_1"].map(self.get_avg_metric)
        output_2_acc = df_experiment_files["output_2"].map(self.get_avg_metric)

        s_acc = (output_0_acc + output_1_acc + output_2_acc)/3

        df_acc = pd.concat([df_experiment_files, s_acc.rename("accuracy")], axis=1)

        # 1. Pivot the data to group by Task on X-axis
        plot_df = df_acc.pivot(index='task', columns=['model', 'framework'], values='accuracy')
        plot_df = plot_df.dropna(axis=1, how='all')

        # 2. Assign styles (Colors for Models, Hatches for Frameworks)
        color_map = {
            'Random': 'lightgray', 
            'Llama-3.1-8B-Instruct': '#318CE7',  
            'Qwen3-32B': '#FF9966'               
        }

        hatch_map = {
            'random': '',      # Solid
            'Single': '',      # Solid
            'Base': '/',        # Diagonal lines
            'Improved': '.'
        }

        original_cols = plot_df.columns
        colors = [color_map[model] for model, framework in original_cols]

        # 3. Plot the base colored bars (edgecolor is required for hatches to show)
        ax = plot_df.plot(
            kind="bar", 
            figsize=(4.5, 3.5), # w x h
            color=colors,
            width=0.85, 
            yticks=np.arange(0, 101, 10), 
            #yticks=s_acc,
            ylabel="Accuracy", 
            rot=0,
            edgecolor='black',
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
            Patch(facecolor=color_map['Qwen3-32B'], edgecolor='black', label='Model: Qwen3-32B'),
            
            # Empty patch for spacing in the legend
            Patch(facecolor='none', edgecolor='none', label=''), 
            
            # Framework patterns (represented with white boxes)
            Patch(facecolor='white', edgecolor='black', hatch='', label='Setup: Single'),
            Patch(facecolor='white', edgecolor='black', hatch='/', label='Setup: Base'),
            Patch(facecolor='white', edgecolor='black', hatch='.', label='Setup: Improved')
        ]

        ax.legend(handles=legend_elements, title = "Model & Setup", loc="upper left")

        plt.tight_layout()
        #plt.show()

        plt.savefig("./visualizations/distributed_base.png")
        plt.close()
        #df_acc.plot(kind="bar", x = "task", y = "accuracy", yticks=np.arange(0, 110, 10), xlabel="Task", ylabel="Accuracy", rot = 0)
        #df_acc.to_csv("./visualizations/distributed_base_results.csv", sep = ";")
        #df_acc.to_latex()