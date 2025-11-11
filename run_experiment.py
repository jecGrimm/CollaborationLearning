from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order
from log import ExperimentLog

logger = ExperimentLog()

cli = CLI()
outfile = cli.experiment_file

if not os.path.isfile(cli.experiment_file):
    # prompt

    if cli.framework == "single":
        single = SingleAgent()
        outfile = single.prompt_model(split = cli.split, device = cli.device) 
    else:
        multi = MultiAgent()
        outfile = multi.prompt_models(split = cli.split, device = cli.device)

# evaluate
evaluate_sent_order(outfile, cli.framework)

log_experiment(outfile)