from __future__ import absolute_import, division, print_function

import logging
import re

import torch
import torch.nn.functional as F

from src.transformers import BertTokenizer, BertForMaskedLM
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


import string
from nltk.corpus import words



import gzip
import json
import random
import warnings
from swords_bert_score.methods.bert import BertInfillingGenerator
from gloss_noise.noise import Gloss_noise
from gloss_score.wordnet import Wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from proposal_score.score import Cmasked
from attention_score.validation_score import ValidationScore
import numpy as np
from tqdm import tqdm
with gzip.open('dataset/SWORDS/test/swords-v1.1_test.json.gz', 'r') as f:
    swords = json.load(f)
    
noise_gloss = Gloss_noise()
wordnet_gloss = Wordnet()
lemmatizer = WordNetLemmatizer()
to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
translate_pos_dict = {"VERB":"V", "NOUN":"N", "ADJ":"J","ADV":"R"}
proposal = Cmasked(128, False, pre_trained="bert-large-uncased")
proposal.get_possible_words()
proposal.get_possible_multitoken_words()
#validation = ValidationScore(128, False, pre_trained="bert-large-uncased")
alpha = 0.05
gamma = 0.5
def clean_context(context, target, target_offset):
    context_offset_marker = [0]*len(context)
    context_offset_marker[target_offset] = 1
    temp_context = ""
    for i in range(len(context)):
        if context[i] in ['\t','\n']:
            temp_context = temp_context + " "
        else:
            temp_context = temp_context + str(context[i])
    
    returned_context = ""
    returned_context_offset_marker = []
    space_before = False
    for i in range(len(temp_context)):
        if temp_context[i] == " ":
            if not space_before:
                returned_context = returned_context + " "
                returned_context_offset_marker = returned_context_offset_marker + [context_offset_marker[i]]
            space_before = True
        else:
            returned_context = returned_context + temp_context[i]
            returned_context_offset_marker = returned_context_offset_marker + [context_offset_marker[i]]
            space_before = False
    
    new_returned_target_offset = -1
    for i in range(len(returned_context)):
        if returned_context_offset_marker[i] == 1:
            new_returned_target_offset = i
    
    assert new_returned_target_offset != -1
    if (returned_context[new_returned_target_offset-1] != " ") and (new_returned_target_offset != 0):
        returned_context = returned_context[:new_returned_target_offset] + " " + returned_context[new_returned_target_offset:]
        new_returned_target_offset += 1

    if (returned_context[new_returned_target_offset+len(target)] != " ") and (new_returned_target_offset+len(target) < len(returned_context)):
        returned_context = returned_context[:(new_returned_target_offset+len(target))] + " " + returned_context[(new_returned_target_offset+len(target)):]
        
    
    if new_returned_target_offset == 0:
        returned_target_index = 0
    else:
        temp = returned_context[:(new_returned_target_offset-1)]
        temp = temp.split(' ')
        returned_target_index = len(temp)
    
    return returned_context, new_returned_target_offset, returned_target_index

generator = BertInfillingGenerator(target_corruption='mask', dropout_p=0.3, top_k=50)

result = {'substitutes_lemmatized': True, 'substitutes': {}}
for tid, target in tqdm(swords['targets'].items()):
    context = swords['contexts'][target['context_id']]
    result['substitutes'][tid] = generator.generate(
        context['context'],
        target['target'],
        target['offset'],
        target_pos=target.get('pos'))
    print(len(result['substitutes'][tid]))
    for i in range(len(result['substitutes'][tid])):
        result['substitutes'][tid][i] = (result['substitutes'][tid][i][0],float(result['substitutes'][tid][i][1]))
print(result)

with open('dataset/SWORDS/test/swords-v1.1_test_original_bert_mask.lsr.json', 'w') as f:
    f.write(json.dumps(result))
