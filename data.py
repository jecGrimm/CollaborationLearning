from datasets import load_dataset
import random
import os

def get_sent_list(batch, i):
    return [batch["sentence1"][i], batch["sentence2"][i], batch["sentence3"][i], batch["sentence4"][i], batch["sentence5"][i]]

class ROCStories:
    def __init__(self, shuffle = False, split = "mini"):
        '''
        This constructor loads the original ROCStories data and creates the shuffled dataset.

        @param
            shuffle: False -> Do not create shuffled data else True
        '''
        self.orig_dataset = load_dataset("csv", data_files="data/ROCStories_winter2017 - ROCStories_winter2017.csv", split="train")

        # load shuffled data
        self.shuffle_file = "./data/ROCStories_shuffled_" + split + ".jsonl"
        if not os.path.exists(self.shuffle_file) or shuffle == True:
            self.shuffled_dataset = self.orig_dataset.map(self.shuffle_sents, batched = True)
            # JSON because of list of shuffled sentences

            if split == "mini":
                self.shuffled_dataset.select(range(3)).to_json(self.shuffle_file, indent=4)
            elif split == "dev":
                self.shuffled_dataset.select(range(1500)).to_json(self.shuffle_file, indent=4)
                #self.shuffled_dataset.select(range(int(0.1 * len(self.orig_dataset)))).to_json(self.shuffle_file, indent=4)
            elif split == "test":
                self.shuffled_dataset.select(range(start = int(0.1 * len(self.orig_dataset)), stop = 1500)).to_json(self.shuffle_file, indent=4)
            else: 
                self.shuffled_dataset.to_json(self.shuffle_file, indent=4) # jsonl because of missing ","
        else:
            self.shuffled_dataset = load_dataset("json", data_files=self.shuffle_file, split="train")

    def shuffle_sents(self, batch):
        '''
        This method shuffles the first four sentences in the instance. 

        @param
            batch: batch of ROCStory instances
        @returns batch_shuffled: original instance with additional shuffled story.
        '''
        batch_shuffled = list()
        for i in range(len(batch["storyid"])):
            ordered_sents = [batch["sentence1"][i], batch["sentence2"][i], batch["sentence3"][i], batch["sentence4"][i], batch["sentence5"][i]]
            ordered_sents_copy = ordered_sents.copy()
            random.shuffle(ordered_sents_copy)
            batch_shuffled.append(ordered_sents_copy)

        return {"shuffled_sents":  batch_shuffled}
        
if __name__ == "__main__":
    roc_data = ROCStories(split = "test")
    print("Done")