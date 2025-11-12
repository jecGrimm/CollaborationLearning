from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order
from log import ExperimentLog

cli = CLI()
config = cli.config
logger = ExperimentLog(config)
outfile = config["experiment_file"]

if not os.path.isfile(outfile):
    # prompt

    if config["framework"] == "single":
        single = SingleAgent()
        
        logger.log_single(single)

        outfile = single.prompt_model(logger, split = cli.split, device = cli.device) 
    else:
        multi = MultiAgent()

        logger.log_multi(multi)

        outfile = multi.prompt_models(split = cli.split, device = cli.device)

# evaluate
evaluate_sent_order(outfile, config["framework"], logger)

logger.store_run_data()