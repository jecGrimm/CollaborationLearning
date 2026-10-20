from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate
from log import ExperimentLog
import torch

def run_experiment(config: dict = None, logger: ExperimentLog = None):
    """
    This function runs the experiment.

    @params
        config: configurations for the experiment
        logger: experiment logger
    """
    # saves computation time
    print(torch.cuda.is_available())
    torch.set_grad_enabled(False)

    if logger == None:
        logger = ExperimentLog(config)

    outfile = config["experiment_file"]

    if not (os.path.isfile(outfile) or os.path.isdir(outfile)):
        # prompt
        if config["framework"] == "single":
            single = SingleAgent(model_id = config["model_id"], split = config["split"], prompt_candidate = config["candidate"], max_new_tokens = config["max_new_tokens"], structured=config["structured"], output_format=config["output_format"], task=config["task"])
            
            logger.log_single(single)

            single.prompt_model(logger, device = config["device"], batch_size = config["batch_size"], thinking_budget=config["thinking_budget"], variation = config["variation"], start = config["start"], end = config["end"], version_idx = config["version_idx"]) 

            outfile = single.define_outfile(logger, hf = True)
        else:
            multi = MultiAgent(model_id = config["model_id"], split = config["split"], conversation_rounds = config["conversation_rounds"], prompt_candidate = config["candidate"], topology=config["topology"], max_new_tokens = config["max_new_tokens"], structured=config["structured"], output_format=config["output_format"], task=config["task"], final = config["final"]) 

            logger.log_multi(multi)

            if config["topology"] == "chain":
                multi.prompt_chain(logger, device = config["device"], batch_size = config["batch_size"], thinking_budget=config["thinking_budget"], variation = config["variation"], final_reorder=config["final_reorder"], conv_format = config["conv_format"], start = config["start"], end = config["end"], version_idx = config["version_idx"])
            elif config["topology"] == "graph":
                multi.prompt_graph(logger, device = config["device"], batch_size = config["batch_size"], thinking_budget=config["thinking_budget"], variation = config["variation"], final_reorder=config["final_reorder"], conv_format = config["conv_format"], start = config["start"], end = config["end"]) # TODO: ADD version_idx
            else:
                multi.prompt_panel(logger, device = config["device"], batch_size = config["batch_size"], thinking_budget=config["thinking_budget"], variation = config["variation"], final_reorder=config["final_reorder"], conv_format = config["conv_format"], start = config["start"], end = config["end"]) # TODO: ADD version_idx

            outfile = multi.define_outfile(logger, hf = True)

    # evaluate 
    evaluate(outfile = outfile, framework = config["framework"], logger = logger, task = config["task"], structured=config["structured"], instances = logger.run_data["instances"])

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
            print(f"Duration: {logger.run_data['duration']}")

        print("\n")
        logger.print_fastest_run(framework=fw, split = split) # testing batch size for efficiency

def test_llm_rewritten_prompts(config: dict):
    """
    This function runs the experiment for different prompts rewritten by other LLMs.

    @params 
        config: experiment configuration
    """
    # file with rewritten prompts
    framework = config["framework"]
    for i, candidate in enumerate(config["candidates"]):
        print("\n")
        print("-" * 30)
        print("Processing candidate: ", candidate)

        logger = ExperimentLog(config)

        if "run_idx" in config.keys():
            logger.run_data["run_idx"] = config["run_idx"][i]

        config["candidate"] = candidate
        run_experiment(config, logger)

    print("\n")
    logger.print_fastest_run(framework=framework, split = config["split"])
    logger.print_best_run(framework = framework, split=config["split"])

if __name__ == "__main__":
    cli = CLI()
    run_experiment(cli.config)
