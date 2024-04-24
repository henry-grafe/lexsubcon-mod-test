from transformers import BertTokenizer, BertForMaskedLM
import torch 
model = BertForMaskedLM.from_pretrained("checkpoint/jjzha_spanbert")
tokenizer = BertTokenizer.from_pretrained("checkpoint/jjzha_spanbert", do_lower_case=False)

model.to("cuda")
max_seq_length = 128

features = tokenizer.encode_plus(
            "This is my first [MASK], so please be gentle with me.",
            add_special_tokens=True,
            max_length=max_seq_length,
            pad_to_max_length='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

input_ids_multimask = features['input_ids']
input_mask_multimask = features['attention_mask']
segment_ids_multimask = features['token_type_ids']

input_ids_multimask = input_ids_multimask.to("cuda")
input_mask_multimask = input_mask_multimask.to("cuda")
segment_ids_multimask = segment_ids_multimask.to("cuda")

input_mask = input_mask_multimask
segment_ids = segment_ids_multimask

print(features)

with torch.no_grad():
            output_multimask = model(input_ids=input_ids_multimask, token_type_ids=segment_ids_multimask, attention_mask=input_mask_multimask)
top_k_words_index = torch.topk(output_multimask[0][0][5], 30)[1].detach().cpu().numpy()
vocab = tokenizer.get_vocab()
vocab = {value: key for key, value in vocab.items()}
for i in range(30):
    print(vocab[top_k_words_index[i]], output_multimask[0][0][5][top_k_words_index[i]].item())
print(len(vocab))
print(len(output_multimask[0][0][5]))