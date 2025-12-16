import re
import torch
from transformers import pipeline, AutoTokenizer, LogitsProcessorList
from data import ROCStories
import os
import time
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_from_disk
from thinking_budget_processor import ThinkingTokenBudgetProcessor
import json

class Agent:
    def __init__(self, model_id: str, split: str, framework: str, interactive = True):
        """
        This class provides functionalities that are shared by the single- and multi-agent frameworks.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            framework: multi or single agent
        """
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.split = split
        self.framework = framework
        self.interactive = interactive
        self.date = str(datetime.now()).replace(" ", "_")

        self.prompt_candidates = json.load(open("./prompts/prompt_candidates.json", 'r', encoding="utf-8"))

    def init_pipe(self, device, logger, thinking_budget):
        if self.framework == "multi":
            if self.interactive:
                max_new_tokens = 256
            else:
                max_new_tokens = int((256 * 11)/6)
        else:
            max_new_tokens = 256 * 11 # give both frameworks the same amount of compute

        if self.model_id == "meta-llama/Llama-3.2-1B-Instruct":
            pipe = pipeline(
                "text-generation",
                model=self.model_id,
                dtype=torch.bfloat16,
                device_map=device,
                max_new_tokens = max_new_tokens
                # cache_dir = "/nfs/gdata/jgrimm/CollaborationLearning/hf_cache/"
            )
            pipe.model.generation_config.pad_token_id = pipe.model.config.eos_token_id[0]
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
            processor = None
        elif "qwen" in self.model_id.lower():
            # ensures that the model stops generating thinking tokens after a while
            processor = ThinkingTokenBudgetProcessor(AutoTokenizer.from_pretrained(self.model_id), max_thinking_tokens=int(max_new_tokens * thinking_budget)) # increased thinking budget
            logger.run_data["thinking_budget"] = processor.max_thinking_tokens

            if processor.max_thinking_tokens > 0:
                # https://huggingface.co/Qwen/Qwen3-14B: Do not use greedy decoding but best thinking configuration
                pipe = pipeline(
                    "text-generation", 
                    model=self.model_id, 
                    dtype=torch.bfloat16,
                    device_map=device,
                    max_new_tokens=int(max_new_tokens-processor.max_thinking_tokens), 
                    logits_processor=LogitsProcessorList([processor]), 
                    temperature = 0.6,
                    top_p = 0.95,
                    top_k = 20,
                    min_p = 0
                )
            else:
                # Temperature=0.7, TopP=0.8, TopK=20, and MinP=0, https://huggingface.co/Qwen/Qwen3-8B
                pipe = pipeline(
                    "text-generation", 
                    model=self.model_id, 
                    dtype=torch.bfloat16,
                    device_map=device,
                    max_new_tokens=max_new_tokens, 
                    logits_processor=LogitsProcessorList([processor]), 
                    temperature = 0.7,
                    top_p = 0.8,
                    top_k = 20,
                    min_p = 0
                )
            
        pipe.tokenizer.padding_side = "left"
        logger.log_pipe(pipe)
    
        return pipe, logger, processor

    def save_outputs(self, logger, outputs, shard_idx: int = None):
        """
        This function saves the model outputs in a json file.

        @warning Huggingface load_dataset cannot load big json files.
        @params
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
            shard_idx: index of the current data shard
            hf: True if output path should lead to a huggingface dataset, False for a json file
        @returns
            path to the output file
        """
        run_idx = logger.get_run_idx()

        output_path = f"./outputs/{self.framework}/{self.model_name}/{self.split}"

        if shard_idx == None:
            if hf == False:
                os.makedirs(output_path, exist_ok = True)
                return f"{output_path}/outputs_{run_idx}.jsonl"
                #return f"./outputs/{self.framework}/{self.model_name}/{self.split}/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}.jsonl"
            else:
                #path = f"./outputs/{self.framework}/{self.model_name}/{self.split}/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}"
                path = f"{output_path}/outputs_{run_idx}"
                os.makedirs(path, exist_ok = True)
                return path
        else:
            if hf == False:
                os.makedirs(f"{output_path}/ckps/", exist_ok = True)
                return f"{output_path}/ckps/outputs_{run_idx}_{shard_idx}.jsonl"
                #os.makedirs(f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/", exist_ok = True)
                #return f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}_{shard_idx}.jsonl"
            else:
                path = f"{output_path}/ckps/outputs_{run_idx}_{shard_idx}"

                #path = f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}_{shard_idx}"
                os.makedirs(path, exist_ok = True)
                return path

class SingleAgent(Agent):
    def __init__(self, model_id: str, split: str, prompt_candidate: str = "adapted-agashe-2025_str_gemini-final"):
        """
        This class defines single-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
        """
        self.model_id = model_id
        self.split = split

        # initialize Agent
        super().__init__(model_id=model_id, split=split, framework="single", interactive = False)

        self.PROMPT = self.prompt_candidates[prompt_candidate]["single"]
        #self.SYSTEM_PROMPT = self.prompt_candidates[prompt_candidate]["system"]

    def create_message(self, batch):
        """
        This method creates a message for a Llama model.

        @params
            batch: batch of instances
        @returns
            dictionary containing the prompt
        """
        batch_messages = []
        for i in range(len(batch["storyid"])):
            #messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
            message = []
            message.append({"role": "user", "content": self.PROMPT.format(shuffled_sents = batch["shuffled_sents"][i])})
            #batch_messages.append(self.PROMPT.format(shuffled_sents = batch["shuffled_sents"][i]))

            # if not thinking:
            #     message.append({"role": "assitant", "content": "<think>\n\n</think>\n\n"})
            
            batch_messages.append(message)

        return {"prompt": batch_messages}
    
    def prompt_model(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation = False):
        """
        This method prompts the model.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, _ = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget) # processor not needed because models are only prompted once
    
        # prepare prompts
        roc_data = ROCStories(split = self.split, variation = variation)
        logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

        prompt_data = roc_data.shuffled_dataset.map(self.create_message, batched=True)
        full_output_list = []
        full_thinking_answer = []

        start = time.time()
        # process data in smaller parts for checkpointing
        num_shards = int((len(prompt_data)/300))+1
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
            for model_out in outputs:
                model_answer = model_out[0]["generated_text"][-1]["content"]
                final_answer = model_answer.split("</think>")[-1].strip()

                output_list.append(final_answer)
                thinking_answer.append(model_answer)

            # output_list = [re.sub(r".*Answer:[ \n]*", "", turn["generated_text"]).split("</think>")[-1].strip() for conv in outputs for turn in conv]
            # thinking_answer = [re.sub(r".*Answer:[ \n]*", "", turn["generated_text"]).split("<think>")[-1].strip() for conv in outputs for turn in conv]

            #output_list = [turn["generated_text"].split("Answer:\n")[-1] for conv in outputs for turn in conv]
            full_output_list += output_list
            full_thinking_answer += thinking_answer

            output_data = shard_data.add_column("output", output_list)
            output_data = shard_data.add_column("thinking_output", thinking_answer)
            
            outpath = self.save_outputs_hf(logger, outputs=output_data, shard_idx=shard_idx)

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})

            logger.store_run_data()

            torch.cuda.empty_cache()


        end = time.time()
        logger.run_data.update({"duration": end - start}) # track processing time

        #print(outputs[0][0]["generated_text"])

        # output_list = [turn["generated_text"] for conv in outputs for turn in conv]
        # num_missing = len(prompt_data) - len(output_list)
        # output_list = output_list + [""] * num_missing

        # save full output
        output_data = prompt_data.add_column("output", full_output_list)

        # model_name = model_id.split("/")[-1]
        # os.makedirs(f"./outputs/single/{model_name}/{split}", exist_ok = True)
        # outfile = f"./outputs/single/{model_name}/{split}/outputs_single_{model_name}_{split}_{str(datetime.now()).replace(" ", "_")}.jsonl"
        # output_data.to_json(outfile, indent=4)
        outpath = self.save_outputs_hf(logger, output_data)
        logger.run_data.update({"instances": len(full_output_list), "experiment_file": outpath})

        logger.store_run_data()

class MultiAgent(Agent):
    def __init__(self, model_id: str, split: str, conversation_rounds: int = 2, prompt_candidate = "adapted-agashe-2025_str_gemini-final", topology = "chain"):
        """
        This class defines multi-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            conversation_rounds: round of conversations
        """
        self.topology = topology

        if self.topology in ["chain", "graph"]:
            self.interactive = True
        else:
            self.interactive = False

        super().__init__(model_id=model_id, split=split, framework="multi", interactive=self.interactive)

        # define prompts
        self.SYSTEM_PROMPT = self.prompt_candidates[prompt_candidate]["system"]

        self.START_PROMPT = self.prompt_candidates[prompt_candidate]["start"]

        self.MID_PROMPT = self.prompt_candidates[prompt_candidate]["mid"] 

        self.RULES = self.prompt_candidates[prompt_candidate]["rules"]

        self.SYSTEM_FINAL = self.prompt_candidates[prompt_candidate]["system_final"]

        self.FINAL = self.prompt_candidates[prompt_candidate]["final"]

        # define player names
        self.NAMES = self.prompt_candidates[prompt_candidate]["names"]
        #self.NAMES = ["Soraya", "Nimrod", "Odett", "Wangui", "Cyrene"]

        self.conversation_rounds = conversation_rounds 

    def create_message(self, conv_round: int, turn: int, conversation: str, roles: dict, player_name: str, final: bool, shuffled_sents: list = None, init_answer = None):
        """
        This method creates the prompt message.

        @params
            conv_round: current conversation round
            turn: current turn in the conversation
            roles: dictionary with the player names and sentences
            player_name: name of the current player
        @returns
            message: list of system and user prompt
        """
        # define agent state
        other_players = [name for name in self.NAMES if name != player_name] 
        
        if not final:
            player_sent = roles[player_name]
            # define system prompt
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

            # define user prompt (start or mid conversation)
            if self.interactive == False or (self.topology == "chain" and conv_round == 0 and turn == 0) or (self.topology == "graph" and conv_round == 0):
                message.append(
                    {
                        "role": "user",
                        "content": self.START_PROMPT.format(
                        player_name = player_name,
                        other_players = other_players,
                        sentence = player_sent
                        )
                    }
                )
            else:
                if init_answer:
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
                else: 
                    message.append(
                        {
                            "role": "user",
                            "content": self.MID_PROMPT.format(
                                player_name = player_name,
                                other_players = other_players,
                                conversation = conversation,
                                sentence = player_sent
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
                        rules = self.RULES,
                        num_players = len(other_players)
                    )
                }
            ]

            if shuffled_sents:
                message.append(
                {
                    "role": "user",
                    "content": self.FINAL.format(
                        shuffled_sents = shuffled_sents,
                        conversation = conversation
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
    
    def create_prompts(self, batch, conv_round: int = -1, turn: int = -1, player_name: str = "", final: bool = False, shuffled_sents: bool = False):
        """
        This method creates the prompts for a batch of instances.

        @params
            batch: batch of instances
            conv_round: iteration in the conversational round
            turn: number of the turn in the conversation
            player_name: name of the current player
        @returns
            dictionary with the prompts for the batch of instances
        """
        messages = []
        for i in range(len(batch["storyid"])):
            if conv_round != 0 and self.topology == "graph" and final == False:
                init_turn, conversation_history = self.get_other_conv(batch["player"][i], batch["turns"][i], turn, conv_round)
            else:
                init_turn = None
                conversation_history = batch["conversation_history"][i]

            if shuffled_sents:
                messages.append(self.create_message(conv_round, turn, conversation_history, batch["roles"][i], player_name, final, shuffled_sents=batch["shuffled_sents"][i], init_answer = init_turn))
            else:
                messages.append(self.create_message(conv_round, turn, conversation_history, batch["roles"][i], player_name, final, init_answer = init_turn))

        return {"prompt": messages}

    def get_conversation_history(self, batch, outputs, player_name: str, final: bool = False):
        """
        This function saves the output in the dataset.

        @params
            batch: batch of instances
            outputs: model outputs
            player_name: name of the current player
        @returns batch: updated batch of instances
        """
        for i, model_out in enumerate(outputs):
            #shard_inst = shard_data.filter(lambda x: x["storyid"] == batch["storyid"][i])

            model_answer = model_out[0]["generated_text"][2]["content"]
            conv_answer = model_answer.split("</think>")[-1].strip()
            conv_answer = re.sub(r"Answer:[ \n]*", "", conv_answer).strip()

            batch["conversation_history"][i] += f"\n{player_name}: {conv_answer}\n"
            batch["turns"][i].append(model_answer)
            batch["player"][i].append(player_name)

            if final == False and "STOP" in model_answer and len(batch["turns"][i]) >= 5: # disable early stopping in first round
                batch["stop"][i] = True # model ended the conversation
        
        return batch

    def update_shard(self, batch, filtered_data, dataset_len):
        """
        This method updates the instances in the shard data with the processed instances.

        @params
            batch: batch of shard data instances
            filtered_data: processed instances
        @returns
            updated batch of instances 
        """
        for i in range(len(batch["storyid"])):
            if batch["stop"][i] == False:
                if len(filtered_data) != dataset_len:
                    processed_inst = filtered_data.filter(lambda x: x["storyid"] == batch["storyid"][i]) # find instance, TODO: make it faster
                else:
                    processed_inst = filtered_data.select([i])

                if len(processed_inst) == 1:
                    batch["conversation_history"][i] = processed_inst["conversation_history"][0]
                    batch["turns"][i] = processed_inst["turns"][0]
                    batch["player"][i] = processed_inst["player"][0] 
                    #batch["roles"][i] = processed_inst["roles"][0] # TODO: check whether role exists
                    batch["stop"][i] = processed_inst["stop"][0]

        return batch

    def prompt_chain(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation = False, final_reorder = True):
        """
        This function prompts the multi-agent framework.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation)
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })

        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/300))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # single processing
            # outputs = shard_data.map(lambda batch: self.play_games(batch, pipe), batched=True, batch_size=batch_size)
            # self.save_outputs_hf(outputs, shard_idx=shard_idx)

            # full_output_list += outputs["output"]

            # batch processing
            # add conversation to data
            shard_data = shard_data.add_column("stop", [False]*len(shard_data))
            shard_data = shard_data.add_column("conversation_history", [""]*len(shard_data))
            shard_data = shard_data.add_column("turns", [[]]*len(shard_data))
            shard_data = shard_data.add_column("player", [[]]*len(shard_data))
            shard_data = shard_data.add_column("roles", [dict()]*len(shard_data))
            shard_data = shard_data.map(self.define_roles, batched=True)

            #filtered_data = shard_data
            conv_round = 0
            while conv_round < self.conversation_rounds:
                turn = 0
                while turn < 5:
                    # filter out early stopping
                    filtered_data = shard_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)

                    if len(filtered_data) > 0:
                        player_name = self.NAMES[turn]
                        
                        filtered_data = filtered_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name, final = False), batched = True)

                        # prompt 
                        turn_outputs = tqdm([out for out in pipe(
                                list(filtered_data["prompt"]),
                                batch_size = batch_size
                            )], desc = "Processing prompts", total = len(filtered_data["prompt"]))

                        # add model answer to conversation history
                        filtered_data = filtered_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name), batched = True)

                        shard_data = shard_data.map(lambda batch: self.update_shard(batch, filtered_data, len(shard_data)), batched = True)

                        # if processor:
                        #     processor.tokens_generated = 0 # allow thinking for the next answer
                        torch.cuda.empty_cache()
                    else:
                        # stop when all conversations have been stopped by a model
                        turn = 5
                        conv_round = self.conversation_rounds

                    # shard_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name), batched = True)

                    # # prompt 
                    # turn_outputs = tqdm([out for out in pipe(
                    #         list(shard_data["prompt"]),
                    #         max_new_tokens=256,
                    #         batch_size = batch_size
                    #     )], desc = "Processing prompts", total = len(shard_data["start_prompt"]))

                    # # add model answer to conversation history
                    # shard_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name))

                    turn += 1
                conv_round += 1
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        #output_data = prompt_data.add_column("output", full_output_list)
        #outpath = self.save_outputs_hf(output_data)
        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.store_run_data()

    def prompt_panel(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation = False, final_reorder = True):
        """
        This function prompts the multi-agent framework.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation)
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })

        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/300))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # single processing
            # outputs = shard_data.map(lambda batch: self.play_games(batch, pipe), batched=True, batch_size=batch_size)
            # self.save_outputs_hf(outputs, shard_idx=shard_idx)

            # full_output_list += outputs["output"]

            # batch processing
            # add conversation to data
            shard_data = shard_data.add_column("stop", [False]*len(shard_data))
            shard_data = shard_data.add_column("conversation_history", [""]*len(shard_data))
            shard_data = shard_data.add_column("turns", [[]]*len(shard_data))
            shard_data = shard_data.add_column("player", [[]]*len(shard_data))
            shard_data = shard_data.add_column("roles", [dict()]*len(shard_data))
            shard_data = shard_data.map(self.define_roles, batched=True)
            #filtered_data = shard_data
            turn = 0
            while turn < 5:
                player_name = self.NAMES[turn]
                
                filtered_data = shard_data.map(lambda batch: self.create_prompts(batch, 0, 0, player_name, final = False, shuffled_sents = False), batched = True)

                # prompt 
                turn_outputs = tqdm([out for out in pipe(
                        list(filtered_data["prompt"]),
                        batch_size = batch_size
                    )], desc = "Processing prompts", total = len(filtered_data["prompt"]))

                # add model answer to conversation history
                shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name, final = False), batched = True)

                torch.cuda.empty_cache()

                turn += 1
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        #output_data = prompt_data.add_column("output", full_output_list)
        #outpath = self.save_outputs_hf(output_data)
        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.store_run_data()

    def prompt_graph(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, variation = False, final_reorder = True):
        """
        This function prompts the multi-agent framework.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation)
        logger.run_data.update({
            "shuffle_file": roc_data.shuffle_file, 
            "batch_size": batch_size, 
            "conversation_rounds": self.conversation_rounds
        })

        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/300))+1
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
                    
                    filtered_data = shard_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name, final = False, shuffled_sents = False), batched = True)

                    # prompt 
                    turn_outputs = tqdm([out for out in pipe(
                            list(filtered_data["prompt"]),
                            batch_size = batch_size
                        )], desc = "Processing prompts", total = len(filtered_data["prompt"]))

                    # add model answer to conversation history
                    shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name, final = False), batched = True)

                    torch.cuda.empty_cache()

                    turn += 1
                
                conv_round += 1
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents = final_reorder)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.store_run_data()

    def define_roles(self, batch):
        for i in range(len(batch["storyid"])):
            batch["roles"][i] = dict(zip(self.NAMES, batch["shuffled_sents"][i]))
        
        return batch

    def get_final_answer(self, shard_data, pipe, batch_size, shuffled_sents: bool = False):
        if shuffled_sents:
            prompt_data = shard_data.map(lambda batch: self.create_prompts(batch, player_name = "Player 6", final = True, shuffled_sents = True), batched = True)
        else:
            prompt_data = shard_data.map(lambda batch: self.create_prompts(batch, player_name = "Player 6", final = True), batched = True)

        final_outputs = tqdm([out for out in pipe(
            list(prompt_data["prompt"]),
            batch_size = batch_size
        )], desc = "Generating final answers", total = len(prompt_data["prompt"]))

        shard_data = shard_data.map(lambda batch: self.get_conversation_history(batch, final_outputs, "Player 6", final = True), batched = True)
        
        return shard_data

    def prompt_final_only(self, logger, device: str = "cpu", batch_size: int = 1, thinking_budget = 0.2, conv_file = "outputs/multi/Qwen3-32B/dev/outputs_88", conv_final = True, variation = False):
        """
        This function prompts the multi-agent framework.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
            thinking_budget: percentage of tokens allowed for "thinking" tokens
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe, logger, processor = self.init_pipe(device = device, logger = logger, thinking_budget = thinking_budget)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split, variation = variation)
        logger.run_data.update({
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
        num_shards = int((len(prompt_data)/300))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)
            
            # get final answer
            shard_data = self.get_final_answer(shard_data, pipe, batch_size, shuffled_sents=True)

            # save ouput
            outpath = self.save_outputs_hf(logger, outputs=shard_data, shard_idx=shard_idx)

            if full_output:
                full_output = concatenate_datasets([full_output, shard_data])
            else:
                full_output = shard_data

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        outpath = self.save_outputs_hf(logger, full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.store_run_data()

    def remove_final(self, batch):
        for i in range(len(batch["storyid"])):
            final_turn = batch["turns"][i][-1].split("</think>")[-1].strip()
            final_turn = re.sub(r"Answer:[ \n]*", "", final_turn).strip()

            final_player = batch["player"][i][-1]

            batch["turns"][i] = batch["turns"][i][:-1]
            batch["player"][i] = batch["player"][i][:-1]

            batch["conversation_history"][i] = batch["conversation_history"][i].replace(f"\n{final_player}: {final_turn}\n", "")

        return batch

    def get_other_conv(self, players, turns, player_turn, conv_round):
        turn_idx = conv_round * len(turns) - len(turns) + player_turn # last guess
        conv_idx = conv_round * len(turns) 
        player = players[turn_idx]


        init_turn = self.remove_thinking(turns[turn_idx])
        
        other_conv = ""
        for i in range(len(turns[:conv_idx])):
            if player != players[i]: # remove all player turns
                other_conv += f"\n{players[i]}: {self.remove_thinking(turns[i])}\n"

        return init_turn, other_conv
    
    def remove_thinking(self, answer):
        cleaned_answer = answer.split("</think>")[-1].strip()
        cleaned_answer = re.sub(r"Answer:[ \n]*", "", cleaned_answer).strip()

        return cleaned_answer


    
if __name__ == "__main__":
    roc_data = ROCStories(split = "dev")
    multi = MultiAgent(model_id = "Qwen/Qwen3-0.6B", split = "dev")

    prompt_data = load_from_disk("outputs/multi/Qwen3-32B/dev/outputs_88")

    prompt_data = prompt_data.map(multi.remove_final, batched=True)