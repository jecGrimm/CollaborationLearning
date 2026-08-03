# CollaborationLearning
This repository contains the experiments for the paper **Can Multi-Agent Systems Synthesize Knowledge through Discussions? Analyzing Argumentative Reasoning in Distributed Knowledge Settings**. In this work, we study the knowledge synthesis abilities of MAS and the impact of argumentative reasoning in distributed knowledge settings on two communication games (BookDiff and StoryCon).

## Environment
``conda create -f env.yml``

To use Llama with restricted access:
1. Request access on huggingface
2. Create Access token with write access to personal space on huggingface
3. type "hf auth login" in terminal and paste token in it

## Data sources
### BookDiff
CMU Booksum (Bamman and Smith, 2013)
| ID	| title	| genre	| summary | 
| ---- | ----| ----- | ---- | 
| 2	| The Plague	| 'Existentialism, Fiction, Absurdist fiction, Novel'	| The text of The Plague is divided into five parts...	| 

### StoryCon
ROCStories (Mostafazadeh et al., 2016)
| storyid |	storytitle |	sentence1 |	sentence2 |	sentence3 |	sentence4 |	sentence5
| ---- | ----| ----- | ---- | ---- | ---- | ---- |
|8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd |	David Drops the Weight |	David noticed he had put on a lot of weight recently. |	He examined his habits to try and figure out the reason. |	He realized he'd been eating too much fast food lately. |	He stopped going to burger places and started a vegetarian diet. |	After a few weeks, he started to feel much better. |

## File Tree
### annotation
Annotated conversation samples.

#### annotated
##### first_iteration
Annotated samples from the first annotation iteration. We use these samples to develop our taxonomy of generated conversational statements.

##### second_iteration
Annotated samples from the second annotation iteration. We use these samples to validate the taxonomy of generated conversational statements (in-domain: Llama-3.1-8B-Instruct, Qwen3-32B; out-of-domain: Qwen3-8B).

##### instances
This folder contains the instances before annotation.

### configs
Experiment configuration files.

### data
#### cmu_booksum
Folder with datasets for CMU Booksum.

#### rocstories
Folder with datasets for ROCStories.

### eval
#### multi
Evaluated data per model and data source for all MAS frameworks.

#### random 
Evaluated data per data source for the random baselines.

#### single
Evaluated data per model and data source for all single agents.

#### csv-files
Output files for each experiment.

### logs
#### runs
Run configurations per run.

#### slurm
Logs from slurm per run.

#### runs.csv
Run configurations for all runs.

### outputs
This folder contains all experiment outputs. It has the same structure as the *eval* folder.

### prompts
#### llm_prompt_rewrite.txt
This file contains the results from prompts rewritten by LLMs.

#### prompt_candidates.json
This file contains all prompts used for the experiments:
| Task |	Setup |	Prompt |
| ---- | ----| ----- | 
| BookDiff | Single |	cmu_single_unstructured |
| BookDiff | Base | cmu_chain_base |
| BookDiff | Improved | cmu_chain_improved |
| BookDiff | Args/Hypo/Full/Reject/Support/Correct/Incorrect | cmu_structured_chain |
| StoryCon | Single |	single_unstructured |
| StoryCon | Base | chain_base |
| StoryCon | Improved | chain_improved |
| StoryCon | Args/Hypo/Full/Reject/Support/Correct/Incorrect | roc_structured_chain |

### visualizations
#### cmu_booksum
Visualizations for BookDiff, including an example output and significance tests.

#### comp_runs
Comparable run configurations.

#### rocstories
Visualizations for StoryCon, incl. Fig. 7 in the paper (micro precision per turn and topology).

The files `args_hypos.png`, `exploration.png`, and `correctness.png` show the subfigures of Fig. 4 in the paper (micro precision per conversation manipulation).

`dist_split.png` shows Fig. 2 (knowledge synthesis performance MAS vs. single).

### cli.py
Creates the configuration object from the configuration file.

### data.py
Performs pre-processing of the data sources for the communication games.

### env.yml
Contains the conda environment specifications.

### helper.py
Script with additional functionalities that are not part of the main experiment.

### ignore_big_files.py
Used to add large files to `.gitignore`.

### LICENSE
MIT-License for use with citations.

### log.py
Creates a logging object to save all experiment configurations.

### prompt.py
Script to prompt the models.

### run_experiment.py
Script to run the full experiment.

### thinking_budget_processor.py
Class that stops the generation of thinking tokens after a specified number of generated thinking tokens.

### visualization.py
Class that creates all experimental visualizations, including the figures in the paper.

## AI usage
- plot_distributed_base(): generate code to improved styling (= added bar labels, hatches, etc.) by Max (Google, LMArena.ai) 
- documentation of the code
- suggestions for prompt formulations

## References
Bamman, David, and Noah A. Smith. “New Alignment Methods for Discriminative Book Summarization.” arXiv:1305.1319. Preprint, arXiv, May 6, 2013. https://doi.org/10.48550/arXiv.1305.1319.

Mostafazadeh, Nasrin, Nathanael Chambers, Xiaodong He, et al. “A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories.” In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, edited by Kevin Knight, Ani Nenkova, and Owen Rambow. Association for Computational Linguistics, 2016. https://doi.org/10.18653/v1/N16-1098.

