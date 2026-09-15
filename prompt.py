import re
import torch
from transformers import pipeline, AutoTokenizer, LogitsProcessorList, BitsAndBytesConfig
from data import ROCStories, CMUBookSum
import os
import time
from datetime import datetime
from tqdm import tqdm
from datasets import concatenate_datasets, load_from_disk
from thinking_budget_processor import ThinkingTokenBudgetProcessor
import json
import ast
from collections import defaultdict
from data import get_sent_list
from transformers.utils import logging
from vllm import LLM, SamplingParams

class Agent:
    def __init__(self, model_id: str, split: str, framework: str, max_new_tokens: int, task: str, interactive: bool = True):
        """
        This class provides functionalities that are shared by the single- and multi-agent frameworks.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            framework: multi or single agent
            max_new_tokens: maximum number of tokens to generate
            task: rocstories or cmu_booksum
            interactive: single/star: False, graph/chain: True
        """
        # surpresses unneccessary warnings
        logging.set_verbosity_error()

        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.split = split
        self.framework = framework
        self.interactive = interactive
        self.date = str(datetime.now()).replace(" ", "_")
        self.task = task

        self.prompt_candidates = json.load(open("./prompts/prompt_candidates.json", 'r', encoding="utf-8"))

        if self.framework == "multi":
            if self.interactive: 
                self.max_new_tokens = max_new_tokens # graph and chain
            else:
                self.max_new_tokens = int((max_new_tokens * 11)/6) # panel
        else:
            self.max_new_tokens = max_new_tokens * 11 # single, give both frameworks the same amount of compute


    def init_pipe(self, device, logger, thinking_budget):
        '''
        This method initializes the Huggingface pipeline for the model.

        @params
            device: device to run the model on
            logger: ExperimentLog object to log the pipeline information
            thinking_budget: float, the fraction of max_new_tokens to be used for thinking tokens
        @returns
            pipe: Huggingface pipeline object
            logger: ExperimentLog object with updated run_data
            processor: ThinkingTokenBudgetProcessor object if the model is Qwen, None otherwise
        '''
        if "llama" in self.model_id.lower():
            pipe = pipeline(
                "text-generation",
                model=self.model_id,
                dtype=torch.bfloat16,
                device_map=device,
                max_new_tokens = self.max_new_tokens,
            )
            pipe.model.generation_config.pad_token_id = pipe.model.config.eos_token_id[0]
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
            processor = None
        elif "qwen" in self.model_id.lower():
            # ensures that the model stops generating thinking tokens after a while
            processor = ThinkingTokenBudgetProcessor(AutoTokenizer.from_pretrained(self.model_id), max_thinking_tokens=int(self.max_new_tokens * thinking_budget)) # increased thinking budget
            logger.run_data["thinking_budget"] = processor.max_thinking_tokens

            if processor.max_thinking_tokens > 0:
                temperature = 0.6
                top_p = 0.95
            else:
                temperature = 0.7
                top_p = 0.8

            if "32" in self.model_id and "booksum" in self.task:
                # https://huggingface.co/Qwen/Qwen3-14B: Do not use greedy decoding but best thinking configuration
                pipe = pipeline(
                    "text-generation", 
                    model=self.model_id, 
                    dtype=torch.bfloat16, # is casted to float 16 anyways
                    device_map=device,
                    max_new_tokens=int(self.max_new_tokens-processor.max_thinking_tokens), 
                    logits_processor=LogitsProcessorList([processor]), 
                    temperature = temperature,
                    top_p = top_p,
                    top_k = 20,
                    min_p = 0,
                    model_kwargs={
                        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    }
                )
                logger.run_data["quantized"] = True
                print("Quantized!")
            else:
                # Temperature=0.7, TopP=0.8, TopK=20, and MinP=0, https://huggingface.co/Qwen/Qwen3-8B
                pipe = pipeline(
                    "text-generation", 
                    model=self.model_id, 
                    dtype=torch.bfloat16,
                    device_map=device,
                    max_new_tokens=self.max_new_tokens, 
                    logits_processor=LogitsProcessorList([processor]), 
                    temperature = temperature,
                    top_p = top_p,
                    top_k = 20,
                    min_p = 0,
                )
                logger.run_data["quantized"] = False
                print("Running without quantization!")
        
        pipe.tokenizer.padding_side = "left"
        logger.log_pipe(pipe)
    
        return pipe, logger, processor
    
    def init_vllm(self, logger):
        quant_models = ["qwen/qwen3-32b"]

        if self.model_id.lower() in quant_models:
            # quantization="bitsandbytes",
            self.llm = LLM(
                model=self.model_id,
                dtype="bfloat16",
                trust_remote_code=True,
                limit_mm_per_prompt={}, # text-only version
                gpu_memory_utilization=0.7, # Von 0.9 auf 0.7 senken, um VRAM-Overhead abzufangen
                max_model_len=32768,          # Kontext testweise halbieren
                quantization="bitsandbytes"
            )
        else:
            self.llm = LLM(
                model=self.model_id,
                dtype="bfloat16",
                trust_remote_code=True,
                limit_mm_per_prompt={}, # text-only version
                gpu_memory_utilization=0.7, # Von 0.9 auf 0.7 senken, um VRAM-Overhead abzufangen
                max_model_len=32768          # Kontext testweise halbieren
            )

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
            #thinking_token_budget=int(self.max_new_tokens * thinking_budget),
        )

        model_config = self.llm.llm_engine.model_config
        logger.log_vllm(model_config, self.sampling_params)

        return logger

    def save_outputs(self, logger, outputs, shard_idx: int = None):
        """
        This function saves the model outputs in a json file.

        @warning Huggingface load_dataset cannot load big json files.
        @params
            logger: ExperimentLog object to log the output file path
            outputs: Huggingface dataset containing the model outputs
            shard_idx: index of the current data shard
        @returns
            outfile: json file containing the output data
        """
        outfile = self.define_outfile(logger, shard_idx=shard_idx)
        outputs.to_json(outfile, indent=4)

        return outfile

    def save_outputs_hf(self, logger, outputs, shard_idx: int = None):
        """
        This function saves the model outputs in a Huggingface directory.

        @params
            logger: ExperimentLog object to log the output file path
            outputs: Huggingface dataset containing the model outputs
            shard_idx: index of the current data shard
        @returns
            outfile: path to the Huggingface dataset
        """
        outpath = self.define_outfile(logger, shard_idx=shard_idx, hf = True)
        outputs.save_to_disk(outpath)

        return outpath

    def define_outfile(self, logger, shard_idx: int = None, hf: bool = False):
        """
        This function returns the name of the output file.

        @params
            logger: ExperimentLog object to log the output file path
            shard_idx: index of the current data shard
            hf: True if output path should lead to a huggingface dataset, False for a json file
        @returns
            path to the output file
        """
        run_idx = logger.get_run_idx()

        output_path = f"./outputs/{self.framework}/{self.model_name}/{self.task}/{self.split}"

        if shard_idx == None:
            if hf == False:
                os.makedirs(output_path, exist_ok = True)
                return f"{output_path}/outputs_{run_idx}.jsonl"
            else:
                path = f"{output_path}/outputs_{run_idx}"
                os.makedirs(path, exist_ok = True)
                return path
        else:
            if hf == False:
                os.makedirs(f"{output_path}/ckps/", exist_ok = True)
                return f"{output_path}/ckps/outputs_{run_idx}_{shard_idx}.jsonl"
            else:
                path = f"{output_path}/ckps/outputs_{run_idx}_{shard_idx}"

                os.makedirs(path, exist_ok = True)
                return path

class SingleAgent(Agent):
    def __init__(self, model_id: str, split: str, prompt_candidate: str = "adapted-agashe-2025_str_gemini-final", max_new_tokens = 256, structured = True, output_format = "full", task = "rocstories"):
        """
        This class defines single-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            prompt_candidate: str with the prompt candidate to use
            max_new_tokens: maximum number of tokens to generate
            structured: boolean indicating whether the output should be structured
            output_format: str indicating the output format, can be "hypo", "full", "support", or "reject"
            task: str indicating the task to perform
        """
        self.model_id = model_id
        self.split = split

        # initialize Agent
        super().__init__(model_id=model_id, split=split, framework="single", task = task, interactive = False, max_new_tokens=max_new_tokens)

        self.PROMPT = self.prompt_candidates[prompt_candidate]["single"]
        self.SYSTEM_PROMPT = self.prompt_candidates[prompt_candidate]["system"]
        self.RULES = self.prompt_candidates[prompt_candidate]["rules"]

        if self.task == "rocstories":
            if output_format == "hypo":
                self.OUTPUT_FORMAT = "[\n{{\n'sentence': '{sentence1}',\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence2}',\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence3}'\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence4}',\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence5}',\n'position': <number of the position>}}\n]" 
            elif output_format == "full":
                self.OUTPUT_FORMAT = "[\n{{\n'sentence': '{sentence1}',\n'argument_1': <reason for the first position>,\n'argument_2': <reason for the second position>,\n'argument_3': <reason for the third position>,\n'argument_4': <reason for the fourth position>,\n'argument_5': <reason for the fifth position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence2}',\n'argument_1': <reason for the first position>,\n'argument_2': <reason for the second position>,\n'argument_3': <reason for the third position>,\n'argument_4': <reason for the fourth position>,\n'argument_5': <reason for the fifth position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence3}'\n'argument_1': <reason for the first position>,\n'argument_2': <reason for the second position>,\n'argument_3': <reason for the third position>,\n'argument_4': <reason for the fourth position>,\n'argument_5': <reason for the fifth position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence4}',\n'argument_1': <reason for the first position>,\n'argument_2': <reason for the second position>,\n'argument_3': <reason for the third position>,\n'argument_4': <reason for the fourth position>,\n'argument_5': <reason for the fifth position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence5}',\n'argument_1': <reason for the first position>,\n'argument_2': <reason for the second position>,\n'argument_3': <reason for the third position>,\n'argument_4': <reason for the fourth position>,\n'argument_5': <reason for the fifth position>,\n'position': <number of the position>}}\n]" 
            elif output_format == "support":
                self.OUTPUT_FORMAT = "[\n{{\n'sentence': '{sentence1}',\n'reason': <reason for the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence2}',\n'reason': <reason for the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence3}'\n'reason': <reason for the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence4}',\n'reason': <reason for the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence5}',\n'reason': <reason for the position>,\n'position': <number of the position>}}\n]" 
            elif output_format == "reject":
                self.OUTPUT_FORMAT = "[\n{{\n'sentence': '{sentence1}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence2}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence3}'\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence4}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence5}',\n'reason': <reason against the position>,\n'position': <number of the position>}}\n]" 
        else:
            if output_format == "hypo":
                self.OUTPUT_FORMAT = "[\n{{\n'title': '{title1}',\n'label': <label odd-one>\n}},\n{{\n'title': '{title2}',\n'label': <label odd-one>\n}},\n{{\n'title': '{title3}',\n'label': <label odd-one>\n}}\n]" 
            elif output_format == "full":
                self.OUTPUT_FORMAT = "[\n{{\n'title': '{title1}',\n'argument_true': <reason for odd-one>,\n'argument_false': <reason against odd-one>,\n'label': <label odd-one>\n}},\n{{\n'title': '{title2}',\n'argument_true': <reason for odd-one>,\n'argument_false': <reason against odd-one>,\n'label': <label odd-one>}},\n{{\n'title': '{title3}',\n'argument_true': <reason for odd-one>,\n'argument_false': <reason against odd-one>,\n'label': <label odd-one>}}\n]" 

        self.structured = structured

    def create_message(self, batch):
        """
        This method creates a message for ROCStories.

        @params
            batch: batch of instances
        @returns
            dictionary containing the prompt
        """
        batch_messages = []
        for i in range(len(batch["storyid"])):
            message = []

            if self.structured:
                message.append({"role": "system", "content": self.SYSTEM_PROMPT.format(rules = self.RULES)})

                sample = self.OUTPUT_FORMAT.format(**{f"sentence{n}": sent for n, sent in enumerate(batch["shuffled_sents"][i], start=1)})
                message.append({"role": "user", "content": self.PROMPT.format(sample = sample, shuffled_sents = batch["shuffled_sents"][i])})
            else:
                message.append({"role": "system", "content": self.SYSTEM_PROMPT.format(rules = self.RULES)})
                message.append({"role": "user", "content": self.PROMPT.format(shuffled_sents = batch["shuffled_sents"][i])})
            
            batch_messages.append(message)

        return {"prompt": batch_messages}
    
    def create_message_booksum(self, batch):
        """
        This method creates a message for CMU Booksum.

        @params
            batch: batch of instances
        @returns
            dictionary containing the prompt
        """
        batch_messages = []
        for i in range(len(batch["id"])):
            message = [] 

            message.append({"role": "system", "content": self.SYSTEM_PROMPT.format(rules = self.RULES)})
            
            input_samples = ""
            for n in range(len(batch["ids"][i])):
                input_samples += f"title{n+1}: {batch['titles'][i][n]}\n"
                input_samples += f"summary{n+1}: {batch['summaries'][i][n]}\n"

            message.append({"role": "user", "content": self.PROMPT.format(input_samples = input_samples)})

            batch_messages.append(message)

        return {"prompt": batch_messages}
    
    def prompt_model(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation = False, start = 0, end = None, version_idx = 0, vllm: bool = True):
        """
        This method prompts the model.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: fraction of max_new_tokens to be used for thinking tokens
            variation: whether to use the variation of the dataset
            start: start index of the dataset
            end: end index of the dataset
            version_idx: index of the dataset version

        @warning
            For VLLM, enabled/disabled thinking is not tested yet!
        """
        # initialize pipeline
        torch.cuda.empty_cache()

        if vllm:
            # VLLM raises an error if this is not set
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            self.init_vllm(logger = logger, thinking_budget=thinking_budget)
        else:
            pipe, logger, _ = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget) # processor not needed because models are only prompted once
    
        # prepare prompts
        if self.task == "rocstories":
            roc_data = ROCStories(split = self.split, variation = variation, start = start, end = end, version_idx = version_idx)
            logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})
            logger.run.config.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

            prompt_data = roc_data.shuffled_dataset.map(self.create_message, batched=True)
        else:
            booksum_data = CMUBookSum(split = self.split, start = start, end = end)
            logger.run_data.update({"arranged_file": booksum_data.arranged_file, "batch_size": batch_size})
            logger.run.config.update({"arranged_file": booksum_data.arranged_file, "batch_size": batch_size})

            prompt_data = booksum_data.arranged_data.map(self.create_message_booksum, batched=True)

        # TODO: test
        if thinking_budget == 0:
            enable_think = False
        else:
            enable_think = True
        
        full_output_list = []
        full_thinking_answer = []

        start = time.time()
        # process data in smaller parts for checkpointing
        num_shards = int((len(prompt_data)/100))+1
        processed_inst = 0 # track number of instances
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards", total = num_shards):
            # process shard
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # prompt model
            outputs = tqdm([out for out in pipe(
                    list(shard_data["prompt"]),
                    batch_size = batch_size
                )], desc = "Processing batches", total = len(list(shard_data["prompt"])))
            
            # store shard output
            output_list = []
            thinking_answer = []
            if vllm:
                outputs = self.llm.chat(messages = list(shard_data["prompt"]), sampling_params = self.sampling_params, chat_template_kwargs={"enable_thinking": enable_think}) # TODO: test thinking

                for output in outputs:
                    #print(f"Antwort: {output.outputs[0].text}\n")
                    model_answer = output.outputs[0].text
                    output_list.append(model_answer.split("</think>")[-1].strip())

                    if thinking_budget > 0.0:
                        thinking_answer.append(model_answer)
            else:
                outputs = tqdm([out for out in pipe(
                        list(shard_data["prompt"]),
                        batch_size = batch_size
                    )], desc = "Processing batches", total = len(list(shard_data["prompt"])))
            
                # store shard output
                for model_out in outputs:
                    model_answer = model_out[0]["generated_text"][-1]["content"]
                    output_list.append(model_answer.split("</think>")[-1].strip())

                    if thinking_budget > 0.0:
                        thinking_answer.append(model_answer)

            full_output_list += output_list
            #full_thinking_answer += thinking_answer

            output_data = shard_data.add_column("output", output_list)
            if thinking_budget > 0.0:
                full_thinking_answer += thinking_answer
                output_data = shard_data.add_column("thinking_output", thinking_answer)
            
            outpath = self.save_outputs_hf(logger, outputs=output_data, shard_idx=shard_idx)

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            logger.run.log({"instances": processed_inst, "experiment_file": outpath})

            logger.store_run_data()

            torch.cuda.empty_cache()

        end = time.time()
        logger.run_data.update({"duration": end - start}) # track processing time

        # save full output
        output_data = prompt_data.add_column("output", full_output_list)

        outpath = self.save_outputs_hf(logger, output_data)
        logger.run_data.update({"instances": len(full_output_list), "experiment_file": outpath})
        logger.run.log({"instances": len(full_output_list), "experiment_file": outpath})

        logger.store_run_data()

class MultiAgent(Agent):
    def __init__(self, model_id: str, split: str, conversation_rounds: int = 2, prompt_candidate: str = "adapted-agashe-2025_str_gemini-final", topology = "chain", max_new_tokens=256, structured = True, output_format = "full", task: str = "rocstories", final: bool = True):
        """
        This class defines multi-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            conversation_rounds: round of conversations
            prompt_candidate: str with the prompt candidate to use
            topology: str with the topology of the conversation, can be "chain", "graph", or "panel"
            max_new_tokens: maximum number of tokens to generate
            structured: boolean indicating whether the output should be structured
            output_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            task: rocstories or cmu_booksum
            final: boolean indicating whether to use the final answer agent
        """
        self.topology = topology

        if self.topology in ["chain", "graph"]:
            self.interactive = True
        else:
            self.interactive = False

        super().__init__(model_id=model_id, split=split, framework="multi", interactive=self.interactive, max_new_tokens=max_new_tokens, task=task)

        # define prompts
        self.SYSTEM_PROMPT = self.prompt_candidates[prompt_candidate]["system"]

        self.START_PROMPT = self.prompt_candidates[prompt_candidate]["start"]

        self.MID_PROMPT = self.prompt_candidates[prompt_candidate]["mid"] 

        self.RULES = self.prompt_candidates[prompt_candidate]["rules"]

        self.use_final = final
        if final:
            self.SYSTEM_FINAL = self.prompt_candidates[prompt_candidate]["system_final"]
            self.FINAL = self.prompt_candidates[prompt_candidate]["final"]

        # define player names
        try:
            self.NAMES = self.prompt_candidates[prompt_candidate]["names"]
        except:
            self.NAMES = None

        self.conversation_rounds = conversation_rounds 

        self.structured = structured

        if self.task == "rocstories":
            self.num_agents = 6 

            self.OUTPUT_FORMAT_HYPO = "{{\n'sentence': '{sentence}',\n'position': <number of the position>\n}}"
            if output_format == "full":
                self.OUTPUT_FORMAT = "{{\n'sentence': '{sentence}',\n'reasoning': '<reasoning about the story>',\n'argument_1': '<argument for the first position>',\n'argument_2': '<argument for the second position>',\n'argument_3': '<argument for the third position>',\n'argument_4': '<argument for the fourth position>',\n'argument_5': '<argument for the fifth position>',\n'position': <number of the position>\n}}"
            elif output_format == "hypo":
                self.OUTPUT_FORMAT = "{{\n'sentence': '{sentence}',\n'position': <number of the position>\n}}"
            elif output_format == "support":
                self.OUTPUT_FORMAT = "{{\n'sentence': '{sentence}',\n'reason': <reason for the position>,\n'position': <number of the position>\n}}" 
            elif output_format == "reject":
                self.OUTPUT_FORMAT = "[\n{{\n'sentence': '{sentence1}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence2}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence3}'\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence4}',\n'reason': <reason against the position>,\n'position': <number of the position>\n}},\n{{\n'sentence': '{sentence5}',\n'reason': <reason against the position>,\n'position': <number of the position>}}\n]" 
        else:
            self.num_agents = 4

            if not self.structured:
                if output_format == "hypo":
                    self.OUTPUT_FORMAT = "[\n{{\n'title': '{title1}',\n'label': '<label odd-one>'\n}},\n{{\n'title': '{title2}',\n'label': '<label odd-one>'\n}},\n{{\n'title': '{title3}',\n'label': '<label odd-one>'\n}}\n]" 
                elif output_format == "full":
                    self.OUTPUT_FORMAT = "[\n{{\n'title': '{title1}',\n'reasoning': '<reasoning about the story>',\n'argument_true': '<argument for odd-one>',\n'argument_false': '<argument against odd-one>',\n'label': '<label odd-one>'\n}},\n{{\n'title': '{title2}',\n'argument_true': '<argument for odd-one>',\n'argument_false': '<argument against odd-one>',\n'label': '<label odd-one>'}},\n{{\n'title': '{title3}',\n'argument_true': '<argument for odd-one>',\n'argument_false': '<argument against odd-one>',\n'label': '<label odd-one>'}}\n]" 
            else:
                if output_format == "full":
                    self.OUTPUT_FORMAT = "{{\n'title': '{title}',\n'reasoning': '<reasoning about the story>',\n'argument_true': '<argument for odd-one>',\n'argument_false': '<argument against odd-one>',\n'label': '<label odd-one>'\n}}"

    def create_message(self, conv_round: int, turn: int, conversation: str, roles: dict, player_name: str, final: bool, shuffled_sents: list = None, titles: list = None, summaries: list = None, init_answer = None):
        """
        This method creates the prompt message.

        @params
            conv_round: current conversation round
            turn: current turn in the conversation
            roles: dictionary with the player names and sentences
            player_name: name of the current player
            final: boolean indicating whether to use the final answer agent
            shuffled_sents: list of shuffled sentences for ROCStories
            titles: list of titles for CMU Booksum
            summaries: list of summaries for CMU Booksum
            init_answer: indicates whether this is the intial answer or a mid-conversation answer
        @returns
            message: list of system and user prompt
        """
        # define agent state
        if self.NAMES:
            other_players  = [name for name in self.NAMES if name != player_name] 
        else:
            other_players = None 
        
        if not final:
            player_sent = ""
            if self.task == "rocstories":
                player_sent = roles[player_name]
            else:
                player_sent = f"title: {roles[player_name][0]}\n"
                player_sent += f"summary: {roles[player_name][1]}\n"
            # define system prompt
            if self.use_final:
                message = [
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT.format(
                            player_name = player_name,
                            other_players = other_players,
                            rules = self.RULES
                            )
                    }
                ]
            else:
                message = [
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT.format(
                            rules = self.RULES
                            )
                    }
                ]

            # define user prompt (start or mid conversation)
            if self.interactive == False or (self.topology == "chain" and conv_round == 0 and turn == 0) or (self.topology == "graph" and conv_round == 0):
                if self.task == "rocstories":
                    # START prompts
                    if self.structured:
                        sample = f"[\n{self.OUTPUT_FORMAT.format(sentence = player_sent)}\n]"
                        message.append(
                            {
                                "role": "user",
                                "content": self.START_PROMPT.format(
                                    sample = sample,
                                    shuffled_sents = f"['{player_sent}']"
                                )
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": self.START_PROMPT.format(
                                    sentence = player_sent
                                )
                            }
                        )
                else:
                    # START prompts
                    if self.structured:
                        sample = f"[\n{self.OUTPUT_FORMAT.format(title = roles[player_name][0])}\n]"
                        message.append(
                            {
                                "role": "user",
                                "content": self.START_PROMPT.format(
                                    sample = sample,
                                    input_samples = player_sent
                                )
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": self.START_PROMPT.format(
                                    input_samples = player_sent
                                )
                            }
                        )
            else:
                # MID prompts
                if init_answer:
                    if self.structured:
                        max_sent_idx = len(shuffled_sents)

                        sample = "[\n"
                        for idx in range(max_sent_idx):
                            sample += self.OUTPUT_FORMAT.format(sentence = shuffled_sents[idx])

                            if idx != max_sent_idx -1:
                                sample += ",\n"
                        sample += "\n]"

                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    other_solutions = conversation,
                                    shuffled_sents = shuffled_sents,
                                    player_solution = init_answer,
                                    sample = sample
                                )
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    player_name = player_name,
                                    other_players = other_players,
                                    other_solutions = conversation,
                                    sentence = player_sent,
                                    player_solution = init_answer
                                )
                            }
                        )
                elif self.structured and not init_answer:
                    if self.task == "rocstories":
                        if conv_round == 0:
                            max_sent_idx = turn + 1
                        else: 
                            max_sent_idx = len(shuffled_sents)

                        sample = "[\n"
                        for idx in range(max_sent_idx):
                            sample += self.OUTPUT_FORMAT.format(sentence = shuffled_sents[idx])

                            if idx != max_sent_idx -1:
                                sample += ",\n"

                        sample += "\n]"
                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    sample = sample,
                                    conversation = conversation,
                                    shuffled_sents = shuffled_sents[:max_sent_idx]
                                )
                            }
                        ) 
                    else:
                        if conv_round == 0:
                            max_title_idx = turn + 1
                        else: 
                            max_title_idx = len(roles)

                        sample = "[\n"
                        for idx in range(max_title_idx):
                            sample += self.OUTPUT_FORMAT.format(title = list(roles.values())[0][0])

                            if idx != max_title_idx -1:
                                sample += ",\n"

                        sample += "\n]"
                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    sample = sample,
                                    conversation = conversation,
                                    input_samples = player_sent
                                )
                            }
                        ) 
                else: 
                    if self.task == "rocstories":
                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    conversation = conversation,
                                    sentence = player_sent
                                )
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": self.MID_PROMPT.format(
                                    conversation = conversation,
                                    input_samples = player_sent
                                )
                            }
                        )
        else:
            # define system prompt
            message = [
                {
                    "role": "system",
                    "content": self.SYSTEM_FINAL.format(
                        player_name = player_name,
                        other_players = other_players,
                        rules = self.RULES                    
                    )
                }
            ]

            if self.task == "cmu_booksum":
                input_samples = ""
                for v in roles.values():
                    input_samples = f"title: {v[0]}\n"
                    input_samples += f"summary: {v[1]}\n"

                message.append(
                    {
                        "role": "user",
                        "content": self.FINAL.format(
                            conversation = conversation,
                            input_samples = input_samples
                        )
                    }
                )

            else:
                if shuffled_sents and not self.structured:
                    message.append(
                    {
                        "role": "user",
                        "content": self.FINAL.format(
                            shuffled_sents = shuffled_sents,
                            conversation = conversation
                            )
                        }
                    )
                elif self.structured:
                    max_sent_idx = len(shuffled_sents)

                    sample = "[\n"
                    for idx in range(max_sent_idx):
                        sample += self.OUTPUT_FORMAT_HYPO.format(sentence = shuffled_sents[idx])

                        if idx != max_sent_idx -1:
                            sample += ",\n"
                    sample += "\n]"

                    message.append(
                        {
                            "role": "user",
                            "content": self.FINAL.format(
                                sample = sample,
                                conversation = conversation,
                                shuffled_sents = shuffled_sents
                            )
                        }
                    )
                else:
                    message.append(
                        {
                            "role": "user",
                            "content": self.FINAL.format(
                                conversation = conversation
                            )
                        }
                    )
        return message
    
    def create_prompts(self, batch, conv_round: int = -1, turn: int = -1, player_name: str = "", final: bool = False, shuffled_sents: bool = False, conv_format: str = "full"):
        """
        This method creates the prompts for a batch of instances.

        @params
            batch: batch of instances
            conv_round: iteration in the conversational round
            turn: number of the turn in the conversation
            player_name: name of the current player
            final: boolean indicating whether to use the final answer agent
            shuffled_sents: boolean indicating whether to use the shuffled sentences
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        @returns
            dictionary with the prompts for the batch of instances
        """
        messages = []

        if self.task == "rocstories":
            for i in range(len(batch["storyid"])):
                if conv_round != 0 and self.topology == "graph" and final == False:
                    init_turn, conversation_history = self.get_other_conv(batch["player"][i], batch["turns"][i], turn, conv_round, conv_format= conv_format)
                else:
                    init_turn = None
                    conversation_history = batch["conversation_history"][i]

                if shuffled_sents:
                    messages.append(self.create_message(conv_round, turn, conversation_history, batch["roles"][i], player_name, final, shuffled_sents=batch["shuffled_sents"][i], init_answer = init_turn))
                else:
                    messages.append(self.create_message(conv_round, turn, conversation_history, batch["roles"][i], player_name, final, init_answer = init_turn))
        else:
            for i in range(len(batch["id"])):
                if conv_round != 0 and self.topology == "graph" and final == False:
                    init_turn, conversation_history = self.get_other_conv(batch["player"][i], batch["turns"][i], turn, conv_round, conv_format= conv_format)
                else:
                    init_turn = None
                    conversation_history = batch["conversation_history"][i]

                messages.append(self.create_message(conv_round, turn, conversation_history, batch["roles"][i], player_name, final, init_answer = init_turn))    

        return {"prompt": messages}

    def get_conversation_history(self, batch, outputs, player_name: str, final: bool = False, conv_format: str = "hypo", vllm: bool = True):
        """
        This function saves the output in the dataset.

        @params
            batch: batch of instances
            outputs: model outputs
            player_name: name of the current player
            final: boolean indicating whether to use the final answer agent
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            vllm: if True, model output is formatted according to vllm, otherwise to huggingface
        
        @returns batch: updated batch of instances
        """
        for i, model_out in enumerate(outputs):
            if conv_format in ["correct", "incorrect"]:
                if self.task == "rocstories":
                    self.gold = get_sent_list(batch, i)
                else:
                    self.gold = batch["gold_title"][i] # correct, incorrect

            if vllm:
                model_answer = model_out.outputs[0].text
            else:
                model_answer = model_out[0]["generated_text"][2]["content"]

            conv_answer, last_hypos = self.create_conv_answer(batch, i, model_answer, conv_format, final)
            
            if player_name.startswith("a ") or player_name.startswith("an "):
                conv_name = player_name.replace("a ", "")
                conv_name = conv_name.replace("an ", "")
                conv_name = conv_name[0].upper() + conv_name[1:]
            else:
                conv_name = player_name

            if conv_answer != "":
                batch["conversation_history"][i] += f"\n{conv_name}: {conv_answer}\n"

            if last_hypos != None:
                batch["last_hypos"][i] = last_hypos.copy() 

            batch["turns"][i].append(model_answer)
            batch["player"][i].append(player_name)

            if final == False and "STOP" in model_answer and len(batch["turns"][i]) >= (self.num_agents-1): # disable early stopping in first round
                batch["stop"][i] = True # model ended the conversation
        
        return batch
    
    def create_conv_answer(self, batch, i, model_answer, conv_format = "full", final=False):
        '''
        This method creates the conversation answer from the model output.

        @params
            batch: batch of instances
            i: index of the instance in the batch
            model_answer: model output
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            final: boolean indicating whether this is the final answer
        @returns
            conv_answer: conversation answer
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

        if not self.structured or final:
            return conv_str, None
        else:
            try:
                conv_list = ast.literal_eval(conv_str)
                if self.task == "rocstories":
                    return self.formulate_conv_roc(batch, i, conv_list, conv_format)
                else: 
                    return self.formulate_conv_cmu(batch, i, conv_list, conv_format) 
            except:
                if self.task == "rocstories":
                    conv_list = self.extract_conv_dict_rocstories(conv_str)
                    return self.formulate_conv_roc(batch, i, conv_list, conv_format)
                else:
                    conv_list = self.extract_conv_dict_cmu(conv_str)
                    return self.formulate_conv_cmu(batch, i, conv_list, conv_format)
                
    def extract_conv_dict_rocstories(self, conv_str: str):
        '''
        This method extracts the conversation dictionary from the model output for ROCStories.

        @params
            conv_str: model output
        @returns
            conv_list: list of dictionaries containing the conversation
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

    def extract_conv_dict_cmu(self, conv_str: str):
        '''
        This method extracts the conversation dictionary from the model output for CMU.

        @params
            conv_str: model output
        @returns
            conv_list: list of dictionaries containing the conversation
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
        This method strips the argument, position, or label from the item.

        @params
            item: string to be stripped
        @returns
            item: stripped string
        '''
        if "argument" in item:
            item = item.strip("argument")
        elif "position" in item:
            item = item.strip("position")
        elif "label" in item:
            item = item.strip("label")
        return item

    def formulate_conv_roc(self, batch, idx, conv_list, conv_format):
        '''
        This method turns the returned dictionary in the conversation answer for ROCStories.

        @params
            batch: batch of instances
            idx: index of the instance in the batch
            conv_list: list of dictionaries containing the conversation
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        @returns
            conv_answer: conversation answer
            new_last_hypos: dictionary with the last hypotheses for each sentence (reused for support and reject formats)
        '''
        new_last_hypos = None
        if conv_format in ["reject", "support"]: 
            prev_last_hypos = {k:v for k,v in batch["last_hypos"][idx].items() if v}.copy() # remove None
            new_last_hypos = {k:v for k,v in batch["last_hypos"][idx].items() if v}.copy()

        conv_answer = ""
        for n, conv_dir in enumerate(conv_list):
            if conv_format == "hypo":
                if conv_dir['position'] != "":
                    conv_answer += f"I propose position {conv_dir['position']} for the sentence '{conv_dir['sentence']}'. {conv_dir['reasoning']}"
                else:
                    conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"

            elif conv_format == "reasons": 
                conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"
                for i in range(1, 6):
                    key = f"argument_{i}"

                    if key in conv_dir.keys():
                        conv_answer += f" {conv_dir[key]}"
            elif conv_format == "support":
                conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"

                prev_hypos = []
                if conv_dir["sentence"] in prev_last_hypos.keys():
                    prev_hypos = prev_last_hypos[conv_dir["sentence"]]

                if prev_hypos == []:
                    for i in range(1, 6): # add all arguments at first
                        key = f"argument_{i}"

                        if key in conv_dir.keys():
                            conv_answer += f" {conv_dir[key]}"
                else:
                    for hypo in prev_hypos: # add rejecting args for all previous hypos
                        hypo_key = f"argument_{hypo}"

                        for k, v in conv_dir.items():
                            if k == hypo_key and k.startswith("argument"):
                                conv_answer += f" {v}"

                if conv_dir["position"] != "":
                    if conv_dir["sentence"] in new_last_hypos.keys() and conv_dir["position"] not in new_last_hypos[conv_dir["sentence"]]:
                        new_last_hypos[conv_dir["sentence"]].append(conv_dir["position"]) # add current hypo as last hypo
                    else:
                        new_last_hypos.update({conv_dir["sentence"]: [conv_dir["position"]]}) # add new first hypo
            elif conv_format == "reject":
                conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"

                prev_hypos = []
                if conv_dir["sentence"] in prev_last_hypos.keys():
                    prev_hypos = prev_last_hypos[conv_dir["sentence"]]

                if prev_hypos == []:
                    for i in range(1, 6): # add all arguments at first
                        key = f"argument_{i}"

                        if key in conv_dir.keys():
                            conv_answer += f" {conv_dir[key]}"
                else:
                    for hypo in prev_hypos: # add rejecting args for all previous hypos
                        hypo_key = f"argument_{hypo}"

                        for k, v in conv_dir.items():
                            if k != hypo_key and k.startswith("argument"):
                                conv_answer += f" {v}"

                if conv_dir["position"] != "":
                    if conv_dir["sentence"] in new_last_hypos.keys() and conv_dir["position"] not in new_last_hypos[conv_dir["sentence"]]:
                        new_last_hypos[conv_dir["sentence"]].append(conv_dir["position"]) # add current hypo as last hypo
                    else:
                        new_last_hypos.update({conv_dir["sentence"]: [conv_dir["position"]]}) # add new first hypo
            elif conv_format == "full": 
                if conv_dir['position'] != "":
                    conv_answer += f"I propose position {conv_dir['position']} for the sentence '{conv_dir['sentence']}'. {conv_dir['reasoning']}"
                else:
                    conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"

                for i in range(1, 6):
                    key = f"argument_{i}"

                    if key in conv_dir.keys():
                        conv_answer += f" {conv_dir[key]}"
            elif conv_format == "correct":
                conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"
                
                if conv_dir["sentence"] in self.gold: 
                    correct_key = f"argument_{self.gold.index(conv_dir['sentence'])+1}"

                    if correct_key in conv_dir.keys():
                        conv_answer += f" {conv_dir[correct_key]}"
            elif conv_format == "incorrect":
                conv_answer += f"Sentence: '{conv_dir['sentence']}'. {conv_dir['reasoning']}"
                
                if conv_dir["sentence"] in self.gold: 
                    correct_key = f"argument_{self.gold.index(conv_dir['sentence'])+1}"

                    for k, v in conv_dir.items():
                        if k != correct_key and k.startswith("argument"):
                            conv_answer += f" {v}"
            if n < len(conv_list) - 1:
                conv_answer += "\n"
        
        return conv_answer, new_last_hypos

    def formulate_conv_cmu(self, batch, idx, conv_list, conv_format):
        '''
        This method turns the returned dictionary in the conversation answer for CMU.

        @params
            batch: batch of instances
            idx: index of the instance in the batch
            conv_list: list of dictionaries containing the conversation
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        @returns
            conv_answer: conversation answer
            new_last_hypos: dictionary with the last hypotheses for each title (reused for support and reject formats)
        '''
        new_last_hypos = None
        if conv_format in ["reject", "support"]: 
            prev_last_hypos = {k:v for k,v in batch["last_hypos"][idx].items() if v}.copy() # remove None
            new_last_hypos = {k:v for k,v in batch["last_hypos"][idx].items() if v}.copy()

        conv_answer = ""
        for n, conv_dir in enumerate(conv_list):
            if conv_format == "hypo":
                if conv_dir['label'].lower() == "true":
                    conv_answer += f"I propose that the book '{conv_dir['title']}' is in a different genre than the other two. {conv_dir['reasoning']}"
                elif conv_dir['label'].lower() == "false":
                    conv_answer += f"I propose that the book '{conv_dir['title']}' is not in a different genre than the other two. {conv_dir['reasoning']}"
                else:
                    conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"
            elif conv_format == "full": 
                if conv_dir['label'].lower() == "true":
                    conv_answer += f"I propose that the book '{conv_dir['title']}' is in a different genre than the other two. {conv_dir['reasoning']}"
                elif conv_dir['label'].lower() == "false":
                    conv_answer += f"I propose that the book '{conv_dir['title']}' is not in a different genre than the other two. {conv_dir['reasoning']}"
                else:
                    conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"

                for i in ["true", "false"]:
                    key = f"argument_{i}"

                    if key in conv_dir.keys():
                        conv_answer += f" {conv_dir[key]}"
            elif conv_format == "reasons": 
                conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"
                for i in ["true", "false"]:
                    key = f"argument_{i}"

                    if key in conv_dir.keys():
                        conv_answer += f" {conv_dir[key]}"
            elif conv_format == "support":
                conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"

                prev_hypos = []
                if conv_dir["title"] in prev_last_hypos.keys():
                    prev_hypos = prev_last_hypos[conv_dir["title"]]

                if prev_hypos == []:
                    for i in ["true", "false"]: # add all arguments at first
                        key = f"argument_{i}"

                        if key in conv_dir.keys():
                            conv_answer += f" {conv_dir[key]}"
                else:
                    for hypo in prev_hypos: # add rejecting args for all previous hypos
                        hypo_key = f"argument_{hypo.lower()}"

                        for k, v in conv_dir.items():
                            if k == hypo_key and k.startswith("argument"):
                                conv_answer += f" {v}"

                if conv_dir["label"] != "":
                    if conv_dir["title"] in new_last_hypos.keys() and conv_dir["label"] not in new_last_hypos[conv_dir["title"]]:
                        new_last_hypos[conv_dir["title"]].append(conv_dir["label"]) # add current hypo as last hypo
                    else:
                        new_last_hypos.update({conv_dir["title"]: [conv_dir["label"]]}) # add new first hypo
            elif conv_format == "reject":
                conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"

                prev_hypos = []
                if conv_dir["title"] in prev_last_hypos.keys():
                    prev_hypos = prev_last_hypos[conv_dir["title"]]

                if prev_hypos == []:
                    for i in ["true", "false"]: # add all arguments at first
                        key = f"argument_{i}"

                        if key in conv_dir.keys():
                            conv_answer += f" {conv_dir[key]}"
                else:
                    for hypo in prev_hypos: # add rejecting args for all previous hypos
                        hypo_key = f"argument_{hypo.lower()}"

                        for k, v in conv_dir.items():
                            if k != hypo_key and k.startswith("argument"):
                                conv_answer += f" {v}"

                if conv_dir["label"] != "":
                    if conv_dir["title"] in new_last_hypos.keys() and conv_dir["label"] not in new_last_hypos[conv_dir["title"]]:
                        new_last_hypos[conv_dir["title"]].append(conv_dir["label"]) # add current hypo as last hypo
                    else:
                        new_last_hypos.update({conv_dir["title"]: [conv_dir["label"]]}) # add new first hypo
            elif conv_format == "correct":
                conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"
                
                if conv_dir["title"] == self.gold:
                    correct_key = f"argument_true"
                else:
                    correct_key = f"argument_false"

                if correct_key in conv_dir.keys():
                    conv_answer += f" {conv_dir[correct_key]}"
            elif conv_format == "incorrect":
                conv_answer += f"Book: '{conv_dir['title']}'. {conv_dir['reasoning']}"
                
                if conv_dir["title"] == self.gold:
                    correct_key = f"argument_true"
                else:
                    correct_key = f"argument_false"

                for k, v in conv_dir.items():
                    if k != correct_key and k.startswith("argument"):
                        conv_answer += f" {v}"
            if n < len(conv_list) - 1:
                conv_answer += "\n"
        
        return conv_answer, new_last_hypos

    def update_shard(self, batch, filtered_data, dataset_len, conv_format):
        """
        This method updates the instances in the shard data with the processed instances.

        @params
            batch: batch of shard data instances
            filtered_data: processed instances
            dataset_len: length of the dataset
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        @returns
            updated batch of instances 
        """
        id_name = ""
        if self.task == "rocstories":
            id_name = "storyid"
        else:
            id_name = "id"

        for i in range(len(batch[id_name])):
            if batch["stop"][i] == False:
                if len(filtered_data) != dataset_len:
                    processed_inst = filtered_data.filter(lambda x: x[id_name] == batch[id_name][i]) # find instance
                else:
                    processed_inst = filtered_data.select([i])

                if len(processed_inst) == 1:
                    batch["conversation_history"][i] = processed_inst["conversation_history"][0]
                    batch["turns"][i] = processed_inst["turns"][0]
                    batch["player"][i] = processed_inst["player"][0] 
                    batch["stop"][i] = processed_inst["stop"][0]

                    if self.structured and conv_format in ["reject", "support"]:
                        batch["last_hypos"][i] = processed_inst["last_hypos"][0]

        return batch

    def prompt_chain(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation: bool = False, final_reorder: bool = True, conv_format: str = "hypo", start: int = 0, end = None, version_idx: int = 0, vllm: bool = True):
        """
        This function prompts the multi-agent framework in a chain topology.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
            variation: boolean indicating whether the ROCStories dataset with or witout variations is used
            thinking_budget: percentage of tokens allowed for "thinking" tokens
            final_reorder: boolean indicating whether to use the final answer agent for reordering
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            start: start index of the dataset
            end: end index of the dataset
            version_idx: index of the ROCStories dataset version
            vllm: if True, runs inference via vllm (offline), otherwise via huggingface
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        if vllm:
            # VLLM raises an error if this is not set
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            logger = self.init_vllm(logger = logger)
            pipe = None
        else:
            pipe, logger, _ = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)

        # prepare dataset
        if self.task == "rocstories":
            roc_data = ROCStories(split = self.split, variation = variation, start = start, end = end, version_idx = version_idx) # TODO: ADD version_idx
            logger.run_data.update({
                "shuffle_file": roc_data.shuffle_file, 
                "batch_size": batch_size, 
                "conversation_rounds": self.conversation_rounds
            })
            logger.run.config.update({
                "shuffle_file": roc_data.shuffle_file, 
                "batch_size": batch_size, 
                "conversation_rounds": self.conversation_rounds
            })

            prompt_data = roc_data.shuffled_dataset
        else:
            cmu_data = CMUBookSum(split = self.split, start = start, end = end)
            logger.run_data.update({
                "arranged_file": cmu_data.arranged_file, 
                "batch_size": batch_size, 
                "conversation_rounds": self.conversation_rounds
            })
            logger.run.config.update({
                "arranged_file": cmu_data.arranged_file, 
                "batch_size": batch_size, 
                "conversation_rounds": self.conversation_rounds
            })

            prompt_data = cmu_data.arranged_data

        if thinking_budget == 0:
            enable_think = False
        else:
            enable_think = True
        
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/100))+1

        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # batch processing
            # add conversation to data
            shard_data = shard_data.add_column("stop", [False]*len(shard_data))
            shard_data = shard_data.add_column("conversation_history", [""]*len(shard_data))
            shard_data = shard_data.add_column("turns", [[]]*len(shard_data))
            shard_data = shard_data.add_column("player", [[]]*len(shard_data))
            shard_data = shard_data.add_column("roles", [dict()]*len(shard_data))
            shard_data = shard_data.map(self.define_roles, batched=True)

            if conv_format in ["reject", "support"]: 
                shard_data = shard_data.add_column("last_hypos", [dict()]*len(shard_data))

            #filtered_data = shard_data
            conv_round = 0
            while conv_round < self.conversation_rounds:
                turn = 0
                while turn < (self.num_agents-1):
                    # filter out early stopping
                    if self.task == "rocstories":
                        filtered_data = shard_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)
                    else:
                        filtered_data = shard_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["id"]))], batched = True)

                    if len(filtered_data) > 0:
                        if self.NAMES:
                            player_name = self.NAMES[turn]
                        else:
                            player_name = f"Player {turn +1}" 
                        
                        filtered_data = filtered_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name, final = False, shuffled_sents = self.structured, conv_format = conv_format), batched = True)

                        if vllm:
                            turn_outputs = self.llm.chat(messages = list(filtered_data["prompt"]), sampling_params = self.sampling_params, chat_template_kwargs={"enable_thinking": enable_think})
                        else:
                            # prompt 
                            turn_outputs = tqdm([out for out in pipe(
                                    list(filtered_data["prompt"]),
                                    batch_size = batch_size
                                )], desc = f"Round {conv_round}, Turn: {turn}", total = len(filtered_data["prompt"]))
                            
                        # add model answer to conversation history
                        filtered_data = filtered_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name, final = False, conv_format = conv_format), batched = True)

                        shard_data = shard_data.map(lambda batch: self.update_shard(batch, filtered_data, len(shard_data), conv_format), batched = True)

                        torch.cuda.empty_cache()
                    else:
                        # stop when all conversations have been stopped by a model
                        turn = self.num_agents-1
                        conv_round = self.conversation_rounds

                    turn += 1
                conv_round += 1
            
            # get final answer
            if self.use_final:
                shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder, conv_format=conv_format) 

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            logger.run.log({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.run.config.update({"experiment_file": outpath}, allow_val_change=True)
        logger.store_run_data()

    def prompt_panel(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation: bool = False, final_reorder: bool = True, conv_format: str = "hypo", start: int = 0, end = None):
        """
        This function prompts the multi-agent framework in a star/panel topology.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
            variation: boolean indicating whether the ROCStories dataset with or witout variations is used
            final_reorder: boolean indicating whether to use the final answer agent with complete information
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            start: start index of the dataset
            end: end index of the dataset

        @warning
            This function is not yet updated to vllm (including thinking).
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation, start = start, end = end) # TODO: ADD version_idx
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })
        logger.run.config.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })


        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/100))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # batch processing
            # add conversation to data
            shard_data = shard_data.add_column("stop", [False]*len(shard_data))
            shard_data = shard_data.add_column("conversation_history", [""]*len(shard_data))
            shard_data = shard_data.add_column("turns", [[]]*len(shard_data))
            shard_data = shard_data.add_column("player", [[]]*len(shard_data))
            shard_data = shard_data.add_column("roles", [dict()]*len(shard_data))
            shard_data = shard_data.map(self.define_roles, batched=True)

            turn = 0
            while turn < 5:
                player_name = self.NAMES[turn]
                
                filtered_data = shard_data.map(lambda batch: self.create_prompts(batch, 0, 0, player_name, final = False, shuffled_sents = False, conv_format = conv_format), batched = True)

                # prompt 
                turn_outputs = tqdm([out for out in pipe(
                        list(filtered_data["prompt"]),
                        batch_size = batch_size
                    )], desc = f"Turn: {turn}", total = len(filtered_data["prompt"]))

                # add model answer to conversation history
                shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name, final = False, conv_format = conv_format), batched = True)

                torch.cuda.empty_cache()

                turn += 1
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder, conv_format = conv_format)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            logger.run.log({"instances": processed_inst, "experiment_file": outpath}, allow_val_change=True)
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.run.config.update({"experiment_file": outpath}, allow_val_change=True)
        logger.store_run_data()

    def prompt_graph(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation: bool = False, final_reorder = True, conv_format: str = "hypo", start: str = 0, end = None):
        """
        This function prompts the multi-agent framework in a graph/layered topology.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
            variation: boolean indicating whether the ROCStories dataset with or witout variations is used
            final_reorder: boolean indicating whether to use the final answer agent with complete information
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            start: start index of the dataset
            end: end index of the dataset

        @warning
            This function is not yet updated to vllm (including thinking).
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation, start = start, end = end) # TODO: ADD version_idx
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })
        logger.run.config.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })

        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/100))+1
        processed_inst = 0
        
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # batch processing
            # add conversation to data
            shard_data = shard_data.add_column("stop", [False]*len(shard_data))
            shard_data = shard_data.add_column("conversation_history", [""]*len(shard_data))
            shard_data = shard_data.add_column("turns", [[]]*len(shard_data))
            shard_data = shard_data.add_column("player", [[]]*len(shard_data))
            shard_data = shard_data.add_column("roles", [dict()]*len(shard_data))
            shard_data = shard_data.map(self.define_roles, batched=True)
            
            conv_round = 0
            while conv_round < self.conversation_rounds:
                turn = 0
                while turn < 5:
                    player_name = self.NAMES[turn]
                    
                    filtered_data = shard_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name, final = False, shuffled_sents = True, conv_format = conv_format), batched = True)

                    # prompt 
                    turn_outputs = tqdm([out for out in pipe(
                            list(filtered_data["prompt"]),
                            batch_size = batch_size
                        )], desc = f"Round {conv_round}, Turn: {turn}", total = len(filtered_data["prompt"]))

                    # add model answer to conversation history
                    shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name, final = False, conv_format = conv_format), batched = True)

                    torch.cuda.empty_cache()

                    turn += 1
                
                conv_round += 1
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder, conv_format = conv_format)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            logger.run.log({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.run.config.update({"experiment_file": outpath}, allow_val_change=True)
        logger.store_run_data()

    def define_roles(self, batch):
        '''
        This method defines the roles of the players in the conversation and assigns the individual information pieces.

        @params
            batch: batch of instances
        @returns
            batch: batch of instances with roles defined
        '''
        if self.task == "rocstories":
            for i in range(len(batch["storyid"])):
                if self.NAMES:
                    batch["roles"][i] = dict(zip(self.NAMES[:-1], batch["shuffled_sents"][i])) 
                else:
                    batch["roles"][i] = dict(zip(["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"], batch["shuffled_sents"][i]))
        else:
            for i in range(len(batch["id"])):
                if self.NAMES:
                    batch["roles"][i] = dict(zip(self.NAMES[:-1], list(zip(batch["titles"][i], batch["summaries"][i])))) 
                else:
                    batch["roles"][i] = dict(zip(["Player 1", "Player 2", "Player 3"], list(zip(batch["titles"][i], batch["summaries"][i]))))
        
        return batch


    def get_final_answer(self, shard_data, pipe, batch_size, shuffled_sents: bool = False, conv_format: str = "hypo", vllm: bool = True):
        '''
        This method prompts the final answer agent to get the final answer for each instance in the shard data.

        @params
            shard_data: shard of instances
            pipe: pipeline for prompting
            batch_size: batch size for prompting
            shuffled_sents: complete information or False
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
            vllm: uses vllm for inference if True, else huggingface
        @returns
            shard_data: shard of instances with final answers
        '''
        if self.NAMES:
            player_name = self.NAMES[-1]
        else:
            player_name = "Player 6"
        
        prompt_data = shard_data.map(lambda batch: self.create_prompts(batch, player_name = player_name, final = True, shuffled_sents = shuffled_sents, conv_format = conv_format), batched = True)

        if vllm:
            final_outputs = self.llm.chat(messages = list(prompt_data["prompt"]), sampling_params = self.sampling_params)
        else:
            final_outputs = tqdm([out for out in pipe(
                list(prompt_data["prompt"]),
                batch_size = batch_size
            )], desc = "Generating final answers", total = len(prompt_data["prompt"]))

        shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, final_outputs, player_name, final = True, conv_format=conv_format), batched = True)
        
        return shard_data

    def prompt_final_only(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, conv_file: str = "outputs/multi/Qwen3-32B/dev/outputs_88", conv_final: bool = True, variation: bool = False, conv_format: str = "hypo"):
        """
        This method creates the final answer for a given conversation history.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
            conv_final: whether conv_file contains conversations with a final answer agent
            conv_file: path to the conversation history file
            variation: boolean indicating whether the ROCStories dataset with or witout variations is used
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        
        @warning
            This function is not yet updated to vllm!
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation) # TODO: ADD version_idx
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conv_file": conv_file,
            "conv_final": conv_final
        })
        logger.run.config.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conv_file": conv_file,
            "conv_final": conv_final
        })

        prompt_data = load_from_disk(conv_file)

        if conv_final:
            prompt_data = prompt_data.map(self.remove_final, batched=True)

        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/100))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents=True, conv_format = conv_format)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            logger.run.log({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.run.config.update({"experiment_file": outpath}, allow_val_change=True)
        logger.store_run_data()

    def remove_final(self, batch):
        '''
        Remove the final turn from each conversation in the batch. Used to create the conversation history without the final answer agent's turn.

        @params
            batch: batch of instances
        @returns
            batch: batch of instances with final turn removed
        '''
        for i in range(len(batch["storyid"])):
            final_turn = batch["turns"][i][-1].split("</think>")[-1].strip()
            final_turn = re.sub(r"Answer:[ \n]*", "", final_turn).strip()

            final_player = batch["player"][i][-1]

            batch["turns"][i] = batch["turns"][i][:-1]
            batch["player"][i] = batch["player"][i][:-1]

            batch["conversation_history"][i] = batch["conversation_history"][i].replace(f"\n{final_player}: {final_turn}\n", "")

        return batch

    def get_other_conv(self, players: list, turns: list, player_turn: int, conv_round: int, conv_format: str = "full"):
        '''
        This method returns the conversation history of all players except the current player.

        @params
            players: list of players in the conversation
            turns: list of turns in the conversation
            player_turn: index of the current player's turn
            conv_round: index of the current conversation round
            conv_format: str indicating the output format, can be "hypo", "full", "reasons", "support", "reject", "correct", or "incorrect"
        @returns
            init_turn: the current player's turn
            other_conv: the conversation history of all other players
        '''
        turn_idx = conv_round * len(turns) - len(turns) + player_turn # last guess
        conv_idx = conv_round * len(turns) 
        player = players[turn_idx]

        init_turn = self.create_conv_answer(turns[turn_idx], conv_format)
        
        other_conv = ""
        for i in range(len(turns[:conv_idx])):
            if player != players[i]: # remove all player turns
                other_conv += f"\n{players[i]}: {self.create_conv_answer(turns[i])}\n"

        return init_turn, other_conv
    
    def remove_thinking(self, answer):
        '''
        This method removes the "thinking" part of the answer and returns only the final answer.

        @params
            answer: the answer string from the model
        @returns
            cleaned_answer: the final answer string without the "thinking" part
        '''
        cleaned_answer = answer.split("</think>")[-1].strip()
        cleaned_answer = re.sub(r"Answer:[ \n]*", "", cleaned_answer).strip()

        return cleaned_answer


    
if __name__ == "__main__":
    roc_data = ROCStories(split = "dev")
    multi = MultiAgent(model_id = "Qwen/Qwen3-0.6B", split = "dev")

    prompt_data = load_from_disk("outputs/multi/Qwen3-32B/dev/outputs_88")

    prompt_data = prompt_data.map(multi.remove_final, batched=True)
