import numpy as np

class Word2VecDistanceComputer():
    def __init__(self) -> None:
        self.target_vocab = open("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_word2vec/embeddings/deps_processed.words.vocab",'r').read().split(" ")
        self.context_vocab = open("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_word2vec/embeddings/deps_processed.contexts.vocab",'r').read().split(" ")
        self.target_embeddings = np.load("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_word2vec/embeddings/deps_processed.words.npy")
        self.context_embeddings = np.load("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_word2vec/embeddings/deps_processed.contexts.npy")
        self.target_word_to_index = {}
        self.context_word_to_index = {}
        for i_target in range(len(self.target_vocab)):
            self.target_word_to_index[self.target_vocab[i_target]] = i_target
        for i_context in range(len(self.context_vocab)):
            self.context_word_to_index[self.context_vocab[i_context]] = i_context
            
distance_computer = Word2VecDistanceComputer()
print(distance_computer.target_word_to_index["and"])