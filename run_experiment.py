from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order
from log import ExperimentLog
import pandas as pd
import torch
import json

def run_experiment(config: dict = None, logger: ExperimentLog = None):
    """
    This function runs the experiment.

    @params
        config: configurations for the experiment
        logger: experiment logger
    """
    # prepare configuration
    if "device" not in config.keys():
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    if logger == None:
        logger = ExperimentLog(config)

    if "device" not in config.keys():
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger["device"] = config["device"]

    outfile = config["experiment_file"]

    if not (os.path.isfile(outfile) or os.path.isdir(outfile)):
        # prompt
        if config["framework"] == "single":
            single = SingleAgent(model_id=config["model_id"], split=config["split"])

            if "user_prompt" in config.keys():
                single.PROMPT = config["user_prompt"]
            
            logger.log_single(single)

            single.prompt_model(logger, device = config["device"], batch_size = config["batch_size"]) 

            outfile = single.define_outfile(hf = True)
        else:
            multi = MultiAgent(model_id=config["model_id"], split=config["split"])

            if "system_prompt" in config.keys():
                multi.SYSTEM_PROMPT = config["system_prompt"]

            if "start_prompt" in config.keys():
                multi.START_PROMPT = config["start_prompt"]

            if "mid_prompt" in config.keys():
                multi.MID_PROMPT = config["mid_prompt"]

            if "rules" in config.keys():
                multi.RULES = config["rules"]

            logger.log_multi(multi)

            multi.prompt_models(logger, device = config["device"], batch_size = config["batch_size"])

            outfile = multi.define_outfile(hf = True)

    # evaluate
    evaluate_sent_order(outfile, config["framework"], logger)

    logger.store_run_data()

def test_batch_size(split: str = "mini", batch_sizes: list = [1, 8, 64, 256, 1000], frameworks: list = ["single", "multi"]):
    """
    This function runs the experiment with different batch sizes.

    @params
        split: data split
        batch_sizes: batch sizes
        frameworks: single- or multi-agent setups
    """
    for fw in frameworks:
        print("\n")
        print("-" * 30)
        print(f"Processing framework {fw}")
        for batch_size in batch_sizes:
            print("\n")
            print(f"Processing batch size {batch_size}")
            config = {
                "split": split,
                "framework": fw,
                "experiment_file": "",
                "batch_size": batch_size,
                "model_id": "meta-llama/Llama-3.2-1B-Instruct"
            }
            logger = ExperimentLog(config) # one logger for each experiment

            run_experiment(config, logger)
            print(f"Duration: {logger.run_data["duration"]}")

        print("\n")
        logger.print_fastest_run(framework=fw, split = split) # testing batch size for efficiency

def test_llm_rewritten_prompts(config: dict):
    """
    This function runs the experiment for different prompts rewritten by other LLMs.

    @params 
        config: experiment configuration
    """
    # file with rewritten prompts
    prompt_candidates = json.load(open("./prompts/prompt_candidates.json", 'r', encoding="utf-8"))
    # prompt_candidates = {
    #     "gemini-2.5-pro": { 
    #         "single":"""\
    #             Arrange the following sentences into a coherent paragraph, ordered by time and logic.\n\
    #             \n\
    #             Sentences to order:\n\
    #             {shuffled_sents}\n\
    #             \n\
    #             Return your answer as a single Python list of strings.\n\
    #             Answer:\n\
    #             """
    #     },
    #     "llama-3.3-70b-instruct": {
    #         "single":'''\
    #         Reorder the following sentences to reflect their logical causal and temporal relationships. Please arrange the sentences in a coherent and chronological order, ensuring that the events described are in a logical sequence. The reordered sentences should be returned as a list in the format:\n\
    #         [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n\
    #         where <sentence1> is the first event in the sequence, <sentence2> is the second event, and so on.\n\
    #         The sentences to be reordered are:\n\
    #         {shuffled_sents}\n\
    #         Please provide the reordered list of sentences.\n\
    #         Answer:\n\
    #         '''
    #     }
    # }

    framework = config["framework"]
    for i, model in enumerate(config["candidates"]):
        print("\n")
        print("-" * 30)
        print("Processing prompt written by ", model)

        # prepare configuration
        if framework == "single":
            config["user_prompt"]= prompt_candidates[model]["single"]
        else:
            config["system_prompt"] = prompt_candidates[model]["system"]
            config["start_prompt"] = prompt_candidates[model]["start"]
            config["mid_prompt"] = prompt_candidates[model]["mid"]
            config["rules"] = prompt_candidates[model]["rules"]

        logger = ExperimentLog(config)
        logger.run_data["run_idx"] = config["run_idx"][i]

        run_experiment(config, logger)

        #print(f"Accuracy of prompt {model}:", logger.run_data["mean_acc"])

    print("\n")
    logger.print_fastest_run(framework=framework, split = config["split"])
    logger.print_best_run(framework = framework, split=config["split"])
        
if __name__ == "__main__":
    cli = CLI()

    #run_experiment(cli.config)
    
    #test_batch_size(split = cli.config["split"], batch_sizes=cli.config["batch_sizes"], frameworks=cli.config["frameworks"])

    #test_batch_size(split = "dev", device = device, batch_sizes=[256], frameworks=["multi"])

    # config = {
    #     "split": "dev",
    #     "device": device,
    #     "framework": "single",
    #     "experiment_file": "",
    #     "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    #     "batch_size": 8
    # }
    # run_experiment(config)

    test_llm_rewritten_prompts(config = cli.config)