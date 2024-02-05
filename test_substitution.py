import torch
from PIL import Image
import open_clip
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

text1 = tokenizer(["I am a man of honor, I keep my word"])
substitutes = ["promise","banana","megabytes","engagements","drip", "part", "word","clothes on", "car"]
text2 = []
for i in range(len(substitutes)):
    text2.append("I am a man of honor, I keep my " + substitutes[i])
text2 = tokenizer(text2)

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features_1 = model.encode_text(text1)
    text_features_2 = model.encode_text(text2)
    
    text_features_1 /= text_features_1.norm(dim=-1, keepdim=True)
    text_features_2 /= text_features_2.norm(dim=-1, keepdim=True)

text_distances = text_features_1 @ text_features_2.T

text_distances = text_distances[0].numpy()
args = np.flip(np.argsort(text_distances))

for i in range(len(substitutes)):
    print(substitutes[args[i]], text_distances[args[i]])
