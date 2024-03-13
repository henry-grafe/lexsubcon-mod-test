from . import vector_operations

def score_candidate_actT(context_vector, target_related_vectors_list, candidate_related_vectors_list, strategy="knn", topk = 10):
    assert strategy in ["knn", "percentage"]
    if strategy == "percentage":
        assert topk < 1.
    
    if ( len(target_related_vectors_list) == 0 ) or ( len(candidate_related_vectors_list) == 0 ):
        return 0.
    
    if strategy == "knn":
        centroid_act_T_s = vector_operations.compute_centroid_of_topk(context_vector, target_related_vectors_list, top_k=topk)
    else:
        centroid_act_T_s = vector_operations.compute_centroid_of_k_percentile(context_vector, target_related_vectors_list, k_percentile=topk)
    
    centroid_P = vector_operations.compute_centroid(candidate_related_vectors_list)
    
    return vector_operations.cosine_similarity(centroid_act_T_s, centroid_P)


def score_candidate_actP(context_vector, target_related_vectors_list, candidate_related_vectors_list, strategy="knn", topk = 10):
    assert strategy in ["knn", "percentage"]
    if strategy == "percentage":
        assert topk < 1.
    
    if ( len(target_related_vectors_list) == 0 ) or ( len(candidate_related_vectors_list) == 0 ):
        return 0.
    
    if strategy == "knn":
        centroid_act_P_s = vector_operations.compute_centroid_of_topk(context_vector, candidate_related_vectors_list, top_k=topk)
    else:
        centroid_act_P_s = vector_operations.compute_centroid_of_k_percentile(context_vector, candidate_related_vectors_list, k_percentile=topk)
    
    centroid_T = vector_operations.compute_centroid(target_related_vectors_list)
    
    return vector_operations.cosine_similarity(centroid_act_P_s, centroid_T)