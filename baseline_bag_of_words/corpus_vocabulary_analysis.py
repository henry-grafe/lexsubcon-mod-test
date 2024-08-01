import pickle
import numpy as np

vocab_characters = pickle.load(file=open("C:\\Users\\NICOLAS\\Documents\\KULeuven\\master_thesis\\datasets\\UKWAC\\UKWAC-2.xml\\UKWAC-2_vocabulary_characters.pickle",'rb'))
print(vocab_characters)
chars = list(vocab_characters.keys())
counts = list(vocab_characters.values())
indexes = np.flip(np.argsort(np.array(counts)))


for i in range(1000):
    print(f"{i}, {chars[indexes[i]]}, {counts[indexes[i]]}")