import gzip
import json
import random
import warnings

with gzip.open('dataset/SWORDS/dev/swords-v1.1_dev.json.gz', 'r') as f:
    swords = json.load(f)

def clean_context(context, target_offset):
    context_offset_marker = [0]*len(context)
    context_offset_marker[target_offset] = 1
    temp_context = ""
    for i in range(len(context)):
        print("we entered the loop")
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
    return returned_context, new_returned_target_offset

def generate(
    context,
    target,
    target_offset,
    target_pos=None):
    
    cleaned_context, new_target_offset = clean_context(context, target_offset)
    print(context)
    print(cleaned_context)
    print(target)
    print(cleaned_context[new_target_offset:(new_target_offset+len(target))])
    if (cleaned_context[(new_target_offset+len(target))] != " " and (new_target_offset != (len(cleaned_context)-1))):
        print("PROBLEEEEEM")
        print(cleaned_context[new_target_offset:(new_target_offset+len(target)+1)])
        #exit(0)
    if (cleaned_context[new_target_offset-1] != " " and (new_target_offset != 0)):
        print("PROBLEEEEEMV2")
        print(cleaned_context[(new_target_offset-1):(new_target_offset+len(target))])
        exit(0)
    cleaned_context.split()
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