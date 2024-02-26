from swords_bert_score.methods.bert import *
import gzip
import json
import random
import warnings
from tqdm import tqdm
import numpy as np

ranker = BertBasedLexSubWithDelemmatizationRanker(compute_validation_score=False)
generator = BertInfillingGenerator(target_corruption="dropout")
def generate(
    context,
    target,
    target_offset,
    target_pos=None):
    
    candidates = generator.generate(context=context, target=target, 
                                    target_offset=target_offset, 
                                    target_pos=target_pos)
        
    scores = []
    words = []
    for word, score in candidates:
        scores.append(float(score))
        words.append(str(word))

    indexes = np.flip(np.argsort(scores))
    k = 10
    top_k = {}
    for i in range(k):
        top_k[words[indexes[i]]] = scores[indexes[i]]
        
    substitutes = list(top_k.keys())
    scores = list(top_k.values())
    #substitutes = ['be', 'have', 'do', 'say', 'getting', 'make', 'go', 'know', 'take', 'see']
    #scores = [random.random() for _ in substitutes]
    return list(zip(substitutes, scores))

with gzip.open('dataset/SWORDS/test/swords-v1.1_test.json.gz', 'r') as f:
    swords = json.load(f)
    
result = {'substitutes_lemmatized': True, 'substitutes': {}}
for tid, target in tqdm(swords['targets'].items()):
    context = swords['contexts'][target['context_id']]
    result['substitutes'][tid] = generate(
        context['context'],
        target['target'],
        target['offset'],
        target_pos=target.get('pos'))

with open('dataset/SWORDS/test/swords-v1.1_test_bert_swords_paper_proposal_score_dropout.lsr.json', 'w') as f:
    f.write(json.dumps(result))