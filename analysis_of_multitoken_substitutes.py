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

test_gold = open("dataset/LS14/test/coinco_test.gold",'r',encoding="latin5").read().split("\n")[:-1]
test_gold_ids = []
for i in range(len(test_gold)):
    test_gold_ids.append(int(test_gold[i].split("::")[0].split(' ')[-2]))

test_gold_reording_indexes = np.argsort(test_gold_ids)
test_gold_reordered = []
for i in range(len(test_gold_reording_indexes)):
    test_gold_reordered.append(test_gold[test_gold_reording_indexes[i]])
print(len(test_gold_reordered))
print(test_gold_reordered)