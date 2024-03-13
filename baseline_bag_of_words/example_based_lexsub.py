from . import vector_operations
from . import text_cleaning
from . import candidate_scorer
from . import corpus_retriever

class ExampleBasedLS():
    def __init__(self) -> None:
        self.text_cleaner = text_cleaning.TextCleaner()
        self.corpus_retriever = corpus_retriever.CorpusRetriever(cleaned_corpus_fp="/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/baseline_bag_of_words/UKWAC-2_processing_ready_retrieval_ready.pickle")
    
    def score_candidates(self, context, target_word, target_id, target_pos, candidate_dict, candidate_pos_dict, strategy="actT"):
        clean_context = self.text_cleaner.clean_context(context=context, target_id=target_id)
        context_vector = vector_operations.vectorize(clean_context)
        clean_target = self.text_cleaner.clean_word(target_word, target_pos)
        related_target_sentences = self.corpus_retriever.find_related_sentences(word_pos=clean_target)
        related_target_vector_list = [vector_operations.vectorize(related_target_sentences[i]) for i in range(len(related_target_sentences))]
        
        candidate_scores = {}
        clean_candidates_list = []
        for candidate_word in list(candidate_dict.keys()):
            clean_candidates_list.append(self.text_cleaner.clean_word(candidate_word, target_pos))
        print("retrieval in process...")
        related_candidates_list_sentences = self.corpus_retriever.find_related_sentences_of_multiple_words(word_pos_list=clean_candidates_list)
        print("retrieval done.")
        c=0
        for candidate_word in list(candidate_dict.keys()):
            related_candidate_sentences = related_candidates_list_sentences[c]
            c+=1
            related_candidate_vector_list = [vector_operations.vectorize(related_candidate_sentences[i]) for i in range(len(related_candidate_sentences))]
            if strategy == "actT":
                candidate_scores[candidate_word] = candidate_scorer.score_candidate_actT(context_vector=context_vector, target_related_vectors_list=related_target_vector_list, 
                                                      candidate_related_vectors_list=related_candidate_vector_list, strategy="knn", topk = 10)
            elif strategy == "actP":
                candidate_scores[candidate_word] = candidate_scorer.score_candidate_actP(context_vector=context_vector, target_related_vectors_list=related_target_vector_list, 
                                                      candidate_related_vectors_list=related_candidate_vector_list, strategy="knn", topk = 10)
            else:
                print("Problem : Unknown strategy, choose \"actT\" or \"actP\"")
        
        return candidate_scores