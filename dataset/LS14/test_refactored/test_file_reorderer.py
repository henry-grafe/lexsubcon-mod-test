import numpy as np
import random
import torch
from proposal_score.score import Cmasked

just_reorder = True
seed = 6809
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

proposal = Cmasked(max_seq_length=128, do_lower_case=True, pre_trained="bert-base-uncased")


test_examples = open("../test/coinco_test.preprocessed",'r',encoding="latin5").read().split("\n")[:-1]
test_examples_ids = []
for i in range(len(test_examples)):
    test_examples_ids.append(int(test_examples[i].split("\t")[1]))

test_examples_reording_indexes = np.argsort(test_examples_ids)
test_examples_reordered = []
for i in range(len(test_examples_reording_indexes)):
    test_examples_reordered.append(test_examples[test_examples_reording_indexes[i]])
test_examples_reordered = test_examples_reordered[:37]+test_examples_reordered[38:]
print(len(test_examples_reordered))



test_gold = open("../test/coinco_test.gold",'r',encoding="latin5").read().split("\n")[:-1]
test_gold_ids = []
for i in range(len(test_gold)):
    test_gold_ids.append(int(test_gold[i].split("::")[0].split(' ')[-2]))

test_gold_reording_indexes = np.argsort(test_gold_ids)
test_gold_reordered = []
for i in range(len(test_gold_reording_indexes)):
    test_gold_reordered.append(test_gold[test_gold_reording_indexes[i]])
print(len(test_gold_reordered))

final_test_examples = []
final_test_gold = []
c=0
for i in range(len(test_examples_reordered)):
    line = test_examples_reordered[i]
    text = line.split("\t")[-1]
    word_index = int(line.split("\t")[-2])
    #print(word_index, text.split(" ")[word_index], text)
    text, target_word_start_index, target_word_end_index, features = proposal.pre_processed_text(text, word_index,
                                                                                             noise_type="GLOSS")
    if target_word_end_index!=target_word_start_index or just_reorder:
        c+=1
        final_test_gold.append(test_gold_reordered[i])
        final_test_examples.append(test_examples_reordered[i])
print(c)
final_test_examples_file = open("coinco_test_reordered.preprocessed", 'w', encoding="latin5")
final_test_gold_file = open("coinco_test_reordered.gold", 'w', encoding="latin5")
for i in range(len(final_test_examples)):
    final_test_examples_file.write(final_test_examples[i]+"\n")
    final_test_gold_file.write(final_test_gold[i] + "\n")
final_test_examples_file.close()
final_test_gold_file.close()