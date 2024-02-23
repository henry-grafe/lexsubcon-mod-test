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

def generate(
    context,
    target,
    target_offset,
    target_pos=None):
    
    cleaned_context, new_target_offset, target_index = clean_context(context, target, target_offset)
    target_pos = translate_pos_dict[target_pos]
    target_word = cleaned_context.split(" ")[target_index]
    
    proposed_words = noise_gloss.created_proposed_list(target_word, wordnet_gloss,
                                                                       target_pos)
    #print(cleaned_context)
    #print(target_word)
    synonyms=[]
    noise_type = "GLOSS"
    if target_word=="":
        word_temp="."
    else:
        word_temp = target_word
    synonyms = noise_gloss.adding_noise(word_temp, wordnet_gloss, target_pos)
    try:
        synonyms.remove(target_word)
    except:
        pass
    
    if len(synonyms)==0 and noise_type == "GLOSS":
        noise_type = "GAUSSIAN"
    if len(proposed_words) > 30:
        pass
    else:
        proposed_words = proposal.proposed_candidates(cleaned_context, target_word, int(target_index), 
                                                      noise_type=noise_type, synonyms=synonyms,
                                                      proposed_words_temp=proposed_words, top_k=30)
        
    lemmatized_word = lemmatizer.lemmatize(target, to_wordnet_pos[target_pos])
    
    try:
        proposed_words.pop(lemmatized_word)
    except:
        pass
    
    main_word = target_word+"."+target_pos
    scores = proposal.predictions(cleaned_context, target_word, main_word, int(target_index),
                                  proposed_words,
                                  noise_type=noise_type, synonyms=synonyms)
    for word in proposed_words:
        proposed_words[word] = proposed_words[word] + alpha * scores[word]
    
    """
    validation.get_contextual_weights_original(cleaned_context, target, target_index, main_word)
    for word in proposed_words:
        text_list = cleaned_context.split(" ")
        text_list[int(target_index)] = word
        text_update = " ".join(text_list)
        validation.get_contextual_weights_update(text_update, word, int(target_index), main_word)
        similarity = validation.get_val_score(word)
        proposed_words[word] = proposed_words[word] + gamma * similarity
    """
    words = []
    scores = []
    for word, score in proposed_words.items():
        words = words + [word]
        scores = scores + [score]
    indexes = np.flip(np.argsort(np.array(scores)))
    top_ten = {}
    for i in range(10):
        top_ten[words[indexes[i]]] = scores[indexes[i]]
    #print("top ten words : ")
    #print(top_ten)
    """       
    substitutes = ['be', 'have', 'do', 'say', 'getting', 'make', 'go', 'know', 'take', 'see']
    scores = [random.random() for _ in substitutes]
    """
    substitutes = list(top_ten.keys())
    scores = list(top_ten.values())
    return list(zip(substitutes, scores))

result = {'substitutes_lemmatized': True, 'substitutes': {}}
for tid, target in tqdm(swords['targets'].items()):
    context = swords['contexts'][target['context_id']]
    result['substitutes'][tid] = generate(
        context['context'],
        target['target'],
        target['offset'],
        target_pos=target.get('pos'))

with open('dataset/SWORDS/test/swords-v1.1_test_proposal_score_val.lsr.json', 'w') as f:
    f.write(json.dumps(result))
