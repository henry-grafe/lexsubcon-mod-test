import pickle
import numpy as np
from copy import deepcopy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 
stop_words = stop_words.union({".",",",":",";","?","!","#","\"","\'","(",")","@card@","I","-","\'s","n\'t","","...","[","]","@ord@","There","|","To","/",""})
unwanted_pos_tags = ['$', "''", '(', ')', ':','IN','LS','MD','PP','TO','VBD','VBN','WDT','WP$','WRB','``','DT','CC','WP']
POS_conversion_list = {'NNS':'N', 'JJ':'J', 'NN':'N', 'NP':'N', 'VVD':'V', 'VVP':'V', 'CD':'N', 'RB':'R', 'VVN':'V', 'VVZ':'V', 'PDT':'N', 'VVG':'V', 'VV':'V', 'JJS':'J', 'JJR':'J', 'RBS':'R', 'RBR':'R', 'RP':'R', 'NPS':'N', 'SYM':'N', 'FW':'N', 'UH':'N'}
class BagOfWordsSubstitution():
    def __init__(self, cleaned_corpus_fp) -> None:
        self.texts = pickle.load(open(cleaned_corpus_fp,'rb'))
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_related_sentences(self, word, pos):
        related_sentences = []
        word_pos = word.lower() + "." + pos
        for i_text in range(len(self.texts)):
            for j_sentences in range(len(self.texts[i_text])):
                if word_pos in self.texts[i_text][j_sentences]:
                    related_sentences.append(self.texts[i_text][j_sentences])
        
        return related_sentences
    
    def compute_normalized_vector(self, word_list, word):
        words = {}
        for w in word_list:
            if w != word:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] += 1
        terms = list(words.keys())
        counts = np.array(list(words.values()),dtype='float')
        counts = counts/np.sqrt((counts**2).sum())
        words = {}
        for i in range(len(counts)):
            words[terms[i]] = counts[i]
        return words
    def compute_centroid(self, normalized_vector_list):
        centroid_vector = {}
        for normalized_vector in normalized_vector_list:
            for word, value in normalized_vector.items():
                if word not in centroid_vector:
                    centroid_vector[word] = value
                else:
                    centroid_vector[word] += value
        for word, value in centroid_vector.items():
            centroid_vector[word] = value/float(len(normalized_vector_list))
        
        terms = list(centroid_vector.keys())
        counts = np.array(list(centroid_vector.values()),dtype='float')
        counts = counts/np.sqrt((counts**2).sum())
        centroid_vector = {}
        for i in range(len(counts)):
            centroid_vector[terms[i]] = counts[i]
        
        return centroid_vector
    
    def compute_similarity(self, norm_vector_1, norm_vector_2):
        total = 0
        for word_1, value_1 in norm_vector_1.items():
            if word_1 in norm_vector_2:
                value_2 = norm_vector_2[word_1]
                total += value_1*value_2
        return total
    
    def score_candidates(self, context, target_word, target_index, target_pos):
        context  =0
        
print("loading corpus...")
substitution = BagOfWordsSubstitution("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_bag_of_words/UKWAC_1_BS4_text_data_cleaned_POS_assembled.pickle")
print("extracting...")

related_N = substitution.extract_related_sentences("fast","J")
for i in range(len(related_N)):
    related_N[i] = substitution.compute_normalized_vector(related_N[i],"fast.J")
related_N = substitution.compute_centroid(related_N)
print(list(related_N.values())[:10])
print((np.array(list(related_N.values()))**2).sum())

related_V = substitution.extract_related_sentences("slow","J")
for i in range(len(related_V)):
    related_V[i] = substitution.compute_normalized_vector(related_V[i],"slow.J")
related_V = substitution.compute_centroid(related_V)
print(len(related_V))

related_W = substitution.extract_related_sentences("snail","N")
for i in range(len(related_W)):
    related_W[i] = substitution.compute_normalized_vector(related_W[i],"snail.N")
related_W = substitution.compute_centroid(related_W)
print(list(related_W.values())[:10])
print((np.array(list(related_W.values()))**2).sum())

print(f"catch.V - wrestle.V = {substitution.compute_similarity(related_V, related_W)}")
print(f"catch.N - wrestle.V = {substitution.compute_similarity(related_N, related_W)}")