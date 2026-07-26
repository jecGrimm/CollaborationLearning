#| filename: thinking_budget_processor.py
#| language: python
# Source: https://muellerzr.github.io/til/end_thinking.html (We modified the code to fit our needs)

from transformers import LogitsProcessor
import torch

class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None, max_total_tokens = 256):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float('-inf')

        self.think_start_token = self.tokenizer.encode("<think>", add_special_tokens=False)[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.think_start_token not in input_ids[0]:
            self.tokens_generated = 0 # new instance, set to 0 again
        else:
            self.tokens_generated += 1 # ongoing instance, count 1 up
        
        for i in range(len(scores)):
            self.stopped_thinking = self.think_end_token in input_ids[i]

            if self.max_thinking_tokens != None and not self.stopped_thinking:

                if self.tokens_generated >= (self.max_thinking_tokens - 1): # after max thinking
                    if self.tokens_generated == self.max_thinking_tokens-1: # insert nl
                        scores[i][:] = self.neg_inf
                        scores[i][self.nl_token] = 0
                    else:
                        scores[i][:] = self.neg_inf 
                        scores[i][self.think_end_token] = 0 # thinking token with very low logs
                        self.stopped_thinking = True

        return scores 
