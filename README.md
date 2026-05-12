# CollaborationLearning
This repository provides an experiment that tests whether single- or multi-agent frameworks gain a deeper conceptual understanding.

## Environment
conda activate collab_learn
Python-Version: 3.13.9

To use Llama with restricted access:
1. Request access on huggingface
2. Create Access token with write access to personal space on huggingface
3. type "hf auth login" in terminal and paste token in it

## Data
ROCStories:
| storyid |	storytitle |	sentence1 |	sentence2 |	sentence3 |	sentence4 |	sentence5
| ---- | ----| ----- | ---- | ---- | ---- | ---- |
|8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd |	David Drops the Weight |	David noticed he had put on a lot of weight recently. |	He examined his habits to try and figure out the reason. |	He realized he'd been eating too much fast food lately. |	He stopped going to burger places and started a vegetarian diet. |	After a few weeks, he started to feel much better. |

Story Cloze Test (SCT)
| InputStoryid	| InputSentence1	| InputSentence2	| InputSentence3	| InputSentence4	| RandomFifthSentenceQuiz1	| RandomFifthSentenceQuiz2	| AnswerRightEnding| 
| ---- | ----| ----- | ---- | ---- | ---- | ---- | ---- |
| 138d5bfb-05cc-41e3-bf2c-fa85ebad14e2	| Rick grew up in a troubled household.	| He never found good support in family, and turned to gangs.	| It wasn't long before Rick got shot in a robbery.	| The incident caused him to turn a new leaf. |	He is happy now.	| He joined a gang.	| 1 | 

## File Tree
### data
Folder with datasets.

### helper.py
Script with additional functionalities that are not part of the main experiment.

### LICENSE
MIT-License for use with citations.

### prompt.py
Script to prompt the models.

## AI usage
plot_distributed_base(): improved styling (= added bar labels, hatches, etc.) by Max (Google, LMArena.ai) 