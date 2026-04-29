from datasets import load_dataset
import random
import os
from collections import defaultdict

def get_sent_list(batch, i):
    return [batch["sentence1"][i], batch["sentence2"][i], batch["sentence3"][i], batch["sentence4"][i], batch["sentence5"][i]]

class ROCStories:
    def __init__(self, shuffle = False, split = "mini", variation = False, start = 0, end = None):
        '''
        This constructor loads the original ROCStories data and creates the shuffled dataset.

        @param
            shuffle: False -> Do not create shuffled data else True
        '''
        if variation:
            self.orig_dataset = load_dataset("csv", data_files="data/rocstories/ROCStories_winter2017 - ROCStories_winter2017.csv", split="train")

            # load shuffled data
            self.shuffle_file = "./data/rocstories/ROCStories_shuffled_" + split + ".jsonl"
        else:
            self.orig_dataset = load_dataset("csv", data_files="data/rocstories/example_good_short_stories - story_modeling_full_scale_123_hits.csv", split="train")

            # load shuffled data
            self.shuffle_file = "./data/rocstories/ROCStories_shuffled_no_variation_" + split + ".jsonl"

        if not end:
            if split == "mini":
                    end = 3
            elif split == "dev":
                if variation:
                    end = int(0.1 * len(self.orig_dataset))
                else:
                    end = 25
            elif split == "test":
                if variation:
                    start = start + int(0.1 * len(self.orig_dataset))
                    end = int(0.1 * len(self.orig_dataset)) + 1500
                else:
                    start = start + 25
                    end = 50

        if not os.path.exists(self.shuffle_file) or shuffle == True:
            self.shuffled_dataset = self.orig_dataset.map(self.shuffle_sents, batched = True)
            # JSON because of list of shuffled sentences

            self.shuffled_dataset.select(range(start, end)).to_json(self.shuffle_file, indent=4)
        else:
            self.shuffled_dataset = load_dataset("json", data_files=self.shuffle_file, split="train")

            self.shuffled_dataset = self.shuffled_dataset.select(range(start, end))

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
        
class CMUBookSum:
    def __init__(self, arrange = False, split = "mini", start = 0, end = None):
        '''
        This constructor loads the original ROCStories data and creates the shuffled dataset.

        @param
            shuffle: False -> Do not create shuffled data else True
        '''
        self.orig_dataset = load_dataset("Iliasselyaa/CMUBookSummaryDataset", split = "train")

        self.split = split

        # load arranged data
        self.arranged_file = "./data/cmu_booksum/CMU_BookSum_arranged_" + split + ".jsonl"
        os.makedirs("./data/cmu_booksum/", exist_ok=True)

        if not os.path.exists(self.arranged_file) or arrange == True:
            if split == "test":
                start = 1000

            if not end:
                if split == "mini":
                    end = 3
                elif split == "dev":
                    end = 1000
                elif split == "test":
                    end = 2000

            self.genre_data = self.orig_dataset.filter(lambda x: x["genres"] != None)
            self.genre_data = self.genre_data.remove_columns(["wikipedia_id", "freebase_id", "author", "publication_date"])
            self.arranged_data = self.genre_data.map(lambda x: self.arrange_data(x))

            self.arranged_data.select(range(start, end)).to_json(self.arranged_file)
        else:
            if not end:
                if split == "mini":
                    end = 3
                elif split == "dev":
                    end = 1000
                elif split == "test":
                    end = 1000

            # TODO: test for null values
            self.arranged_data = load_dataset("json", data_files=self.arranged_file, split="train")

            self.arranged_data = self.arranged_data.select(range(start, end))

    def arrange_data(self, instance):
        genres = instance["genres"].split(", ")

        # find instance with same genre
        found = False
        i = 0
        while not found:
            same_genre = genres[i].lower()
            same_insts = self.genre_data.filter(lambda x: same_genre in x["genres"].lower())

            if len(same_insts) > 0:
                found = True
            else:
                i += 1

        rand_idx = random.randint(0, len(same_insts)-1)
        same_inst = same_insts.select([rand_idx])[0]

        # find instance with different genre
        odd_insts = self.genre_data.filter(lambda x: same_genre not in x["genres"].lower())

        odd_rand_idx = random.randint(0, len(odd_insts)-1)
        odd_inst = odd_insts.select([odd_rand_idx])[0]

        orig_pos = [0, 1, 2] # 0: same inst, 1: same inst, 2: odd inst
        #ids_orig = [instance["id"], same_inst["id"], odd_inst["id"]]
        random.shuffle(orig_pos)
        arranged_ids = []
        arranged_titles = []
        arranged_genres = []
        arranged_summaries = []
        for pos in orig_pos:
            if pos == 0:
                arranged_ids.append(instance["id"])
                arranged_titles.append(instance["title"])
                arranged_genres.append(instance["genres"])
                arranged_summaries.append(instance["summary"])
            elif pos == 1:
                arranged_ids.append(same_inst["id"])
                arranged_titles.append(same_inst["title"])
                arranged_genres.append(same_inst["genres"])
                arranged_summaries.append(same_inst["summary"])
            else:
                arranged_ids.append(odd_inst["id"])
                arranged_titles.append(odd_inst["title"])
                arranged_genres.append(odd_inst["genres"])
                arranged_summaries.append(odd_inst["summary"])

        return {"ids": arranged_ids, "gold_title": odd_inst["title"], "titles": arranged_titles, "same_genre": same_genre, "genres": arranged_genres, "summaries": arranged_summaries}


if __name__ == "__main__":
    # rocdata = ROCStories(shuffle=False, split = "mini", variation = True, start=0, end=3)
    # print(len(rocdata.orig_dataset))
    #outpath = "outputs/random/cmu_booksum/test"

    # roc_data.shuffled_dataset.save_to_disk(outpath)

    #good_inst = roc_data.orig_dataset.filter(lambda x: x["storytitle"] == "Earthquake!")

    book_data = CMUBookSum(arrange=False, split = "mini")
    print(len(book_data.orig_dataset))
    # book_data.arranged_data.save_to_disk(outpath)

    print("Done")