from baseline_bag_of_words.example_based_lexsub import ExampleBasedLS


import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm

from reader import Reader_lexical
from metrics.evaluation import evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    # --------------- train dataset
    parser.add_argument("-tf", "--train_file", type=str, help="path of the train file dataset",
                        default='dataset/LS07/trial/lst_trial.preprocessed')
    parser.add_argument("-gf", "--train_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS07/trial/lst_trial.gold')

    # --------------- test dataset
    parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                        default='dataset/LS07/test/lst_test.preprocessed')
    parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS07/test/lst_test.gold')



    # --------------- output results
    parser.add_argument("-outp", "--output_results", type=str, help="path of the output file with results",
                        default='dataset/results/bow_ls/results_gap_')
    parser.add_argument("-eval", "--results_file", type=str, help="path of the output file with gap metric",
                        default='dataset/results/bow_ls/results_final_gap_')

    # -------------------parameters
    
    parser.add_argument("-fna", "--fna", type=str, help="name to seperate write file",
                        default='firsattempt')
    
    # ----------gap flags
    parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)",
                        default=True)
    parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
                        default='dataset/LS07/lst.gold.candidates')

    parser.add_argument("-seed", "--seed", type=int, help="the seed that we are using for the experiment",
                        default=6809)
    args = parser.parse_args()
    seed = args.seed

    """
    reader of features/labels and candidates if gap
    """
    reader = Reader_lexical()
    reader.create_feature(args.test_file)
    
    bow_ls = ExampleBasedLS()
    
    proposal_flag = True
    if args.gap:
        reader.create_candidates(args.golden_file_cadidates)

    evaluation_metric = evaluation()


    "======================================================================================================"
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    "======================================================================================================"

    count_gloss = 0
    iter_index = 0
    not_found = {}
    final2 = {}
    for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:

                change_word = context[0]
                target_pos = main_word.split('.')[-1]
                text = context[1]
                original_text = text
                index_word = context[2]

                change_word = text.split(' ')[int(index_word)]
                synonyms = []

                if args.gap:
                    try:
                        proposed_words = reader.candidates[main_word]
                    except:
                        # for ..N in LS14
                        proposed_words = ["..", ".", ",", "!"]
                    proposed_words = reader.created_dict_proposed(proposed_words)
                
                
                proposed_words = bow_ls.score_candidates(text, change_word, index_word, target_pos.upper(), proposed_words, {}, strategy="actT")
                keys = list(proposed_words.keys())
                values = list(proposed_words.values())
                indexes = np.flip(np.argsort(values))
                proposed_words = {}
                for i in range(len(indexes)):
                    proposed_words[keys[indexes[i]]] = values[indexes[i]]
                print(main_word, text)
                print(proposed_words)
                
                if args.gap:
                    evaluation_metric.write_results(args.output_results + args.fna + "_" + str(seed) + "_gap.txt",
                                                    main_word, instance,
                                                    proposed_words)
                evaluation_metric.write_results(args.output_results + args.fna + "_" + str(seed) + "_probabilites.txt",
                                                main_word,
                                                instance,
                                                proposed_words)
                evaluation_metric.write_results_p1(args.output_results + args.fna + "_" + str(seed) + "_p1.txt",
                                                   main_word, instance,
                                                   proposed_words)
                evaluation_metric.write_results_lex_oot(args.output_results + args.fna + "_" + str(seed) + ".oot",
                                                        main_word, instance,
                                                        proposed_words, limit=10)
                evaluation_metric.write_results_lex_best(args.output_results + args.fna + "_" + str(seed) + ".best",
                                                         main_word, instance,
                                                         proposed_words, limit=1)

    end = (time.time() - start_time)
    evaluation_metric.write_time(args.output_results + args.fna + "_" + str(seed) + "_time.txt", end)
    if args.gap:
        evaluation_metric.gap_calculation(args.test_golden_file,
                                          args.output_results + args.fna + "_" + str(seed) + "_gap.txt",
                                          args.results_file + args.fna + "_" + str(seed) + "_gap.txt")
    evaluation_metric.calculation_perl(args.test_golden_file,
                                       args.output_results + args.fna + "_" + str(seed) + ".best",
                                       args.output_results + args.fna + "_" + str(seed) + ".oot",
                                       args.results_file + args.fna + "_" + str(seed) + ".best",
                                       args.results_file + args.fna + "_" + str(seed) + ".oot")
    evaluation_metric.calculation_p1(args.test_golden_file,
                                     args.output_results + args.fna + "_" + str(seed) + "_p1.txt",
                                     args.results_file + args.fna + "_" + str(seed) + "_p1.txt")
