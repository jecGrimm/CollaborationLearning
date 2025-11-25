import torch
from transformers import pipeline, AutoTokenizer, LogitsProcessorList
from data import ROCStories
import os
import time
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from thinking_budget_processor import ThinkingTokenBudgetProcessor

class Agent:
    def __init__(self, model_id: str, split: str, framework: str):
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
        self.date = str(datetime.now()).replace(" ", "_")


    def save_outputs(self, outputs, shard_idx: int = None):
        """
        This function saves the model outputs in a json file.

        @warning Huggingface load_dataset cannot load big json files.
        @params
            outputs: Huggingface dataset containing the model outputs
            shard_idx: index of the current data shard
        @returns
            outfile: json file containing the output data
        """
        outfile = self.define_outfile(shard_idx=shard_idx)
        outputs.to_json(outfile, indent=4)

        return outfile

    def save_outputs_hf(self, outputs, shard_idx: int = None):
        """
        This function saves the model outputs in a Huggingface directory.

        @params
            outputs: Huggingface dataset containing the model outputs
            shard_idx: index of the current data shard
        @returns
            outfile: path to the Huggingface dataset
        """
        outpath = self.define_outfile(shard_idx=shard_idx, hf = True)
        outputs.save_to_disk(outpath)

        return outpath

    def define_outfile(self, shard_idx: int = None, hf: bool = False):
        """
        This function returns the name of the output file.

        @params
            shard_idx: index of the current data shard
            hf: True if output path should lead to a huggingface dataset, False for a json file
        @returns
            path to the output file
        """
        if shard_idx == None:
            if hf == False:
                os.makedirs(f"./outputs/{self.framework}/{self.model_name}/{self.split}/", exist_ok = True)
                return f"./outputs/{self.framework}/{self.model_name}/{self.split}/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}.jsonl"
            else:
                path = f"./outputs/{self.framework}/{self.model_name}/{self.split}/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}"
                os.makedirs(path, exist_ok = True)
                return path
        else:
            if hf == False:
                os.makedirs(f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/", exist_ok = True)
                return f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}_{shard_idx}.jsonl"
            else:
                path = f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}_{shard_idx}"
                os.makedirs(path, exist_ok = True)
                return path

class SingleAgent(Agent):
    def __init__(self, model_id: str, split: str):
        """
        This class defines single-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
        """
        self.model_id = model_id
        self.split = split

        # initialize Agent
        super().__init__(model_id=model_id, split=split, framework="single")

        self.PROMPT = '''\
            Please bring the following sentences in the correct order based on their causal and temporal relations:\n\
                {shuffled_sents}\n\
            Return a list of the sentences in the correct order in the following format:\n\
                [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n\
            Answer:\n\
            '''

    def create_message(self, batch):
        """
        This method creates a message for a Llama model.

        @params
            batch: batch of instances
        @returns
            dictionary containing the prompt
        """
        messages = []
        for i in range(len(batch["storyid"])):
            #messages.append({"role": "user", "content": prompt.format(shuffled_sents = batch["shuffled_sents"][i])})
            messages.append(self.PROMPT.format(shuffled_sents = batch["shuffled_sents"][i]))

        return {"prompt": messages}
    
    def prompt_model(self, logger, device: str = "cpu", batch_size: int = 1):
        """
        This method prompts the model.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
        """
        # initialize pipeline
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.bfloat16,
            device_map=device
        )
        pipe.model.generation_config.pad_token_id = pipe.model.config.eos_token_id[0]
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
        pipe.tokenizer.padding_side = "left"
        logger.log_pipe(pipe)
    
        # prepare prompts
        roc_data = ROCStories(split = self.split)
        logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

        prompt_data = roc_data.shuffled_dataset.map(self.create_message, batched=True)
        full_output_list = []

        start = time.time()
        # process data in smaller parts for checkpointing
        num_shards = int((len(prompt_data)/1000))+1
        processed_inst = 0 # track number of instances
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards", total = num_shards):
            # process shard
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)

            # prompt model
            outputs = tqdm([out for out in pipe(
                    list(shard_data["prompt"]),
                    max_new_tokens=256,
                    batch_size = batch_size
                )], desc = "Processing batches", total = len(list(shard_data["prompt"])))
            
            # store shard output
            output_list = [turn["generated_text"].split("Answer:\n")[-1] for conv in outputs for turn in conv]
            full_output_list += output_list

            output_data = shard_data.add_column("output", output_list)
            
            outpath = self.save_outputs_hf(outputs=output_data, shard_idx=shard_idx)

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst, "experiment_file": outpath})

            logger.store_run_data()


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
        outpath = self.save_outputs_hf(output_data)
        logger.run_data.update({"instances": len(full_output_list), "experiment_file": outpath})

        logger.store_run_data()

class MultiAgent(Agent):
    def __init__(self, model_id: str, split: str, conversation_rounds: int = 2):
        """
        This class defines multi-agents.

        @params
            model_id: str with the model_id from Huggingface
            split: data split
            conversation_rounds: round of conversations
        """
        super().__init__(model_id=model_id, split=split, framework="multi")

        # define prompts
        self.SYSTEM_PROMPT = '''
        I am {player_name}, playing a game with my partners {other_players}. The game has the following rules: {rules}. At each time step, I will provide you with the relevant information of the game. I will also provide you with the conversation history up to this point. Please help me to formulate an adequate response to the conversation. If all five sentences have been collected during the conversation, assemble them in the correct order, include "STOP" in your answer and return a list of the sentences in the correct order in the following format:  
        [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]
        '''

        self.START_PROMPT = '''It is currently my ({player_name}) turn and my sentence is "{sentence}". I start the conversation with the other players ({other_players}). Please formulate a beginning statement for the conversation for me. Do not say anything else. Got it?
        Answer:
        '''
        self.MID_PROMPT='''It is currently my ({player_name}) turn and my sentence is "{sentence}". I continue the conversation with the other players ({other_players}).
        The conversation history: 
        {conversation}
        Please formulate a response to the conversation for me. Do not say anything else. Got it?
        Answer:
        '''

        self.RULES = '''
        - The game is played by 5 players.
        - Each player receives one sentence that the other players do not know.
        - All sentences together form a short story and are causally and temporally related to each other.
        - The goal of the game is to reassemble the original story by bringing all five sentences into the correct order.
        - The players conduct a conversation where they exchange their sentences and discuss how they could be related to each other and how they could be ordered correctly.
        
        My sentence: "{sentence}"
        '''

        # define player names
        self.NAMES = ["Soraya", "Nimrod", "Odett", "Wangui", "Cyrene"]

        self.conversation_rounds = conversation_rounds 

    def create_message(self, conv_round: int, turn: int, conversation: str, roles: dict, player_name: str):
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
        player_sent = roles[player_name]
        rules = self.RULES.format(sentence = player_sent)

        # define system prompt
        message = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format(
                    player_name = player_name,
                    other_players = other_players,
                    rules = rules
                    )
            }
        ]

        # define user prompt (start or mid conversation)
        if conv_round == 0 and turn == 0:
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
            while conv_round < 4:
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
                    if "STOP" in model_answer:
                        conv_round = 4
                        turn = 5
                    else: 
                        turn += 1
                conv_round += 1
            
            #batch_convs.append(turns)
            batch_turns.append(turns)
            batch_players.append(players)

        return {"turns": batch_turns, "players": batch_players}
    
    def create_prompts(self, batch, conv_round: int, turn: int, player_name: str):
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
            if conv_round == 0 and turn == 0:
                batch["roles"][i] = dict(zip(self.NAMES, batch["shuffled_sents"][i])) #assing sentences to players

            messages.append(self.create_message(conv_round, turn, batch["conversation_history"][i], batch["roles"][i], player_name))

        return {"prompt": messages}

    def get_conversation_history(self, batch, outputs, player_name: str):
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

            # shard_inst["conversation_history"][0] += f"\n{player_name}: {model_answer}\n"
            # shard_inst["turns"][0].append(model_answer)
            # shard_inst["player"][0].append(player_name)
            # shard_inst["roles"][0] = batch["roles"][i]
            batch["conversation_history"][i] += f"\n{player_name}: {conv_answer}\n"
            batch["turns"][i].append(model_answer)
            batch["player"][i].append(player_name)

            if "STOP" in model_answer:
                batch["stop"][i] = True # model ended the conversation
        
        return batch

    def update_shard(self, batch, filtered_data):
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
                processed_inst = filtered_data.filter(lambda x: x["storyid"] == batch["storyid"][i]) # find instance

                if len(processed_inst) == 1:
                    batch["conversation_history"][i] = processed_inst["conversation_history"][0]
                    batch["turns"][i] = processed_inst["turns"][0]
                    batch["player"][i] = processed_inst["player"][0]
                    batch["roles"][i] = processed_inst["roles"][0]
                    batch["stop"][i] = processed_inst["stop"][0]

        return batch

    def prompt_models(self, logger, device: str = "cpu", batch_size: int = 1):
        """
        This function prompts the multi-agent framework.

        @params
            logger: experiment logger
            device: cpu/cuda
            batch_size: batch size
        """
        # initialize pipeline
        torch.cuda.empty_cache()
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            dtype=torch.bfloat16,
            device_map=device,
            max_new_tokens=256
            # cache_dir = "/nfs/gdata/jgrimm/CollaborationLearning/hf_cache/"
        )
        if self.model_id == "meta-llama/Llama-3.2-1B-Instruct":
            pipe.model.generation_config.pad_token_id = pipe.model.config.eos_token_id[0]
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
        elif "qwen" in self.model_id.lower():
            processor = ThinkingTokenBudgetProcessor(AutoTokenizer.from_pretrained(self.model_id), max_thinking_tokens=int(256*0.1))
            # https://huggingface.co/Qwen/Qwen3-14B: Do not use greedy decoding but best thinking configuration
            pipe = pipeline(
                "text-generation", 
                model=self.model_id, 
                max_new_tokens=int(256-256*0.1), 
                logits_processor=LogitsProcessorList([processor]), 
                device = device,
                temperature = 0.6,
                top_p = 0.95,
                top_k = 20,
                min_p = 0
            )
        pipe.tokenizer.padding_side = "left"
        logger.log_pipe(pipe)
        logger.run_data["thinking_budget"] = int(256*0.1)
        
        # prepare dataset
        roc_data = ROCStories(split = self.split)
        logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

        prompt_data = roc_data.shuffled_dataset
        full_output = None

        start = time.time()
        # process shards
        num_shards = int((len(prompt_data)/1000))+1
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
            #filtered_data = shard_data
            conv_round = 0
            while conv_round < self.conversation_rounds:
                turn = 0
                while turn < 5:
                    # filter out early stopping
                    filtered_data = shard_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)

                    if len(filtered_data) > 0:
                        player_name = self.NAMES[turn]
                        
                        filtered_data = filtered_data.map(lambda batch: self.create_prompts(batch, conv_round, turn, player_name), batched = True)

                        # prompt 
                        turn_outputs = tqdm([out for out in pipe(
                                list(filtered_data["prompt"]),
                                batch_size = batch_size
                            )], desc = "Processing prompts", total = len(filtered_data["prompt"]))

                        # add model answer to conversation history
                        filtered_data = filtered_data.map(lambda batch: self.get_conversation_history(batch, turn_outputs, player_name), batched = True)

                        shard_data = shard_data.map(lambda batch: self.update_shard(batch, filtered_data), batched = True)

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

            # save ouput
            outpath = self.save_outputs_hf(outputs=shard_data, shard_idx=shard_idx)

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
        outpath = self.save_outputs_hf(full_output)
        logger.run_data.update({"experiment_file": outpath})
        logger.store_run_data()
