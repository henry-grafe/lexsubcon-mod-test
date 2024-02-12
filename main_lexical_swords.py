import gzip
import json
import random
import warnings

with gzip.open('dataset/SWORDS/dev/swords-v1.1_dev.json.gz', 'r') as f:
    swords = json.load(f)

def clean_context(context):
    returned_context = ""
    for i in range(len(context)):
        print("we entered the loop")
        if context[i] in ['\t','\n']:
            returned_context = returned_context + " "
        else:
            returned_context = returned_context + str(context[i])
    return returned_context

def generate(
    context,
    target,
    target_offset,
    target_pos=None):
    
    cleaned_context = clean_context(context)
    print(context)
    print(cleaned_context)
    print(target)
    print(cleaned_context[target_offset:(target_offset+len(target))])
    inp = input("next")
    if inp=="q":
        exit(0)
    substitutes = ['be', 'have', 'do', 'say', 'getting', 'make', 'go', 'know', 'take', 'see']
    scores = [random.random() for _ in substitutes]
    return list(zip(substitutes, scores))

result = {'substitutes_lemmatized': True, 'substitutes': {}}
for tid, target in swords['targets'].items():
    context = swords['contexts'][target['context_id']]
    result['substitutes'][tid] = generate(
        context['context'],
        target['target'],
        target['offset'],
        target_pos=target.get('pos'))

with open('dataset/SWORDS/dev/swords-v1.1_dev_mygenerator.lsr.json', 'w') as f:
    f.write(json.dumps(result))