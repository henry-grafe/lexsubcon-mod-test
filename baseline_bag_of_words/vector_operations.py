import numpy as np

def vectorize(word_list):
    word_vector = {}
    for i_word in range(len(word_list)):
        word = word_list[i_word]
        if word not in word_vector:
            word_vector[word] = 1
        else:
            word_vector[word] += 1
    
    terms = list(word_vector.keys())
    counts = np.array(list(word_vector.values()),dtype='float')
    counts = counts/np.sqrt((counts**2).sum())
    word_vector = {}
    for i in range(len(counts)):
        word_vector[terms[i]] = counts[i]
    
    return word_vector

def cosine_similarity(norm_vector_1, norm_vector_2):
    """
    CAUTION : BOTH VECTORS HAVE TO BE NORMALIZED TO EUCLID-NORM = 1.
    """
    total = 0
    for word_1, value_1 in norm_vector_1.items():
        if word_1 in norm_vector_2:
            value_2 = norm_vector_2[word_1]
            total += value_1*value_2
    return total

def cosine_similarity_index_ranking(anchor_word_vector, word_vector_list):
    """
    CAUTION : ALL VECTORS HAVE TO BE NORMALIZED TO EUCLID-NORM = 1.
    """
    similarities = np.zeros(len(word_vector_list), dtype='float')
    for i_vector in range(len(word_vector_list)):
        similarities[i_vector] = cosine_similarity(anchor_word_vector, word_vector_list[i_vector])
    index_rank = np.flip(np.argsort(similarities))
    return index_rank

def compute_centroid(normalized_vector_list):
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

def compute_centroid_of_topk(anchor_word_vector, normalized_vector_list, top_k):
    """
    CAUTION : ALL VECTORS HAVE TO BE NORMALIZED TO EUCLID-NORM = 1.
    """
    index_rank = cosine_similarity_index_ranking(anchor_word_vector, normalized_vector_list)
    
    if top_k > len(normalized_vector_list):
        top_k = len(normalized_vector_list)
    
    top_k_word_vector_list = [normalized_vector_list[index_rank[i]] for i in range(top_k)]
    return compute_centroid(top_k_word_vector_list)

def compute_centroid_of_k_percentile(anchor_word_vector, normalized_vector_list, k_percentile):
    """
    CAUTION : ALL VECTORS HAVE TO BE NORMALIZED TO EUCLID-NORM = 1.
    """
    index_rank = cosine_similarity_index_ranking(anchor_word_vector, normalized_vector_list)
    top_k = int(float(len(normalized_vector_list))*k_percentile)
    if top_k == 0:
        top_k = 1
    if top_k > len(normalized_vector_list):
        top_k = len(normalized_vector_list)
    
    top_k_word_vector_list = [normalized_vector_list[index_rank[i]] for i in range(top_k)]
    return compute_centroid(top_k_word_vector_list)

def cosine_similarity_between_lists(normalized_vector_list_1, normalized_vector_list_2):
    """
    CAUTION : ALL VECTORS HAVE TO BE NORMALIZED TO EUCLID-NORM = 1.
    """
    centroid_vector_1 = compute_centroid(normalized_vector_list_1)
    centroid_vector_2 = compute_centroid(normalized_vector_list_2)
    return cosine_similarity(centroid_vector_1, centroid_vector_2)